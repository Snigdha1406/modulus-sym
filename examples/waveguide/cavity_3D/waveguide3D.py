# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import warnings
from sympy import Symbol, pi, sin, Number, Eq, And

import modulus.sym
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_3d import Box
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.utils.io.plotter import InferencerPlotter
from modulus.sym.key import Key
from modulus.sym.eq.pdes.electromagnetic import PEC, SommerfeldBC, MaxwellFreqReal

x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    # params for domain
    components = {
        "Glass1": {
            "box": Box(point_1=(0, -1, -1), point_2=(5, 1, 1)),  # point_1=(0, -1, -1), point_2= (5, 1, 1)
            "permittivity": 0.375,
            "permeability": 1,
        },
        "Dielectric": {
            "box": Box(point_1=(-0.5, -0.9, -0.5), point_2=(0.5, 0.9, 0.5)),
            "permittivity": 85,
            "permeability": 1,
        },
        "Glass2": {
            "box": Box(point_1=(-5, -1, -1),point_2= (0, 1, 1)),
            "permittivity": 0.375,
            "permeability": 1,
        },
        "Air": {
            "box": Box(point_1=(-0.5, -1, -1),point_2=(10, 1, 1)),
            "permittivity": 1,
            "permeability": 1,
        },
    }

    # Define the domain and add the components
    domain = Domain()
    for name, props in components.items():
        domain.add_component(
            name, props["box"], props["permittivity"], props["permeability"]
        )

    # Define the boundary and interior constraints
    boundary_constraints = [
                               PointwiseBoundaryConstraint(
                                   Key("ElectricField", x, y, z, direction=dir),
                                   0,
                                   domain.get_boundary(name, dir),
                               )
                               for name in components.keys()
                               for dir in ["x_min", "x_max", "y_min", "y_max"]
                           ] + [
                               BlochBoundary(
                                   Key("ElectricField", x, y, z),
                                   Number(-pi / 4),
                                   domain.get_boundary("Air", "z_min"),
                               ),
                               BlochBoundary(
                                   Key("ElectricField", x, y, z),
                                   Number(pi / 4),
                                   domain.get_boundary("Air", "z_max"),
                               ),
                           ]
    interior_constraints = [
        PointwiseInteriorConstraint(
            Key("ElectricField", x, y, z),
            0,
            domain.get_interior(name),
        )
        for name in components.keys()
    ]

    # Define the PDE system
    pde_system = []
    for name in components.keys():
        pde_system.append(
            MaxwellFreqReal(
                Key("ElectricField", x, y, z, component=name),
                Key("MagneticField", x, y, z, component=name),
                domain.get_permittivity(name),
                domain.get_permeability(name),
            )
        )
        # Apply Bloch boundary conditions for Xmin, Xmax, Ymin, Ymax
        for dir in ["x_min", "x_max", "y_min", "y_max"]:
            pde_system.append(
                PointwiseBoundaryConstraint(
                    Key("ElectricField", x, y, z, direction=dir),
                    Key("ElectricField", x, y, z, direction=dir, shift=(1, 0, 0)),
                    domain.get_boundary(name, dir),
                )
            )
            pde_system.append(
                PointwiseBoundaryConstraint(
                    Key("ElectricField", x, y, z, direction=dir),
                    Key("ElectricField", x, y, z, direction=dir, shift=(0, 1, 0)),
                    domain.get_boundary(name, dir),
                )
            )
        # Apply Radiant boundary conditions for zmin and zmax
        for dir in ["z_min", "z_max"]:
            pde_system.append(
                PointwiseBoundaryConstraint(
                    Key("ElectricField", x, y, z, direction=dir),
                    0,
                    domain.get_boundary(name, dir),
                )
            )
        pde_system.append(
            PEC(
                Key("ElectricField", x, y, z, component=name),
                domain.get_boundary(name),
            )
        )
        pde_system.append(
            SommerfeldBC(
                Key("ElectricField", x, y, z, component=name),
                Key("MagneticField", x, y, z, component=name),
                domain.get_boundary(name, "z_max"),
            )
        )
    # Define the solver and solve the problem
    solver = Solver(domain, pde_system)
    solver.solve()

    # Define the inferencer and plot the results
    inferencer = PointwiseInferencer(domain)
    plotter = InferencerPlotter(inferencer)
    plotter.plot(Key("ElectricField", x, y, z))
    plotter.plot(Key("MagneticField", x, y, z))

    # Define numpy inferencer with interior points
    interior_points = domain.sample_interior(
        10000, bounds={x: (0, width), y: (0, length), z: (0, height)}
    )
    numpy_inference = PointwiseInferencer(
        domain=domain,
        invar=interior_points,
        output_names=["ElectricField", "MagneticField"],
        plotter=InferencerPlotter(),
        batch_size=2048,
    )

    import numpy as np
    import matplotlib.pyplot as plt
    from modulus.eq.pdes.electromagnetic import SParameter

    # Define the frequency range
    freq_range = np.linspace(10e12, 20e12, num=100)

    # Calculate S-parameters at different frequencies
    s_params = []
    for freq in freq_range:
        s_params.append(
            SParameter(
                domain, pde_system, freq, ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
            ).solve()
        )

    # Extract transmission from the S-parameters
    transmission = []
    for s_param in s_params:
        s11, s21, s12, s22 = s_param
        transmission.append(np.abs(s21) ** 2 / (1 - np.abs(s11) ** 2))

    # Plot transmission vs frequency
    fig, ax = plt.subplots()
    ax.plot(freq_range / 1e12, transmission)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Transmission (magnitude)")
    ax.set_title("Transmission vs Frequency")
    plt.show()


if __name__ == "__main__":
    run()
