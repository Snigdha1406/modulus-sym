# from sympy import Symbol, pi, sin, Number, Eq, And
#
# import modulus
# from modulus.hydra import instantiate_arch, ModulusConfig
# from modulus.solver import Solver
# from modulus.domain import Domain
# from modulus.geometry.primitives_3d import Box
# from modulus.domain.constraint import (
#     PointwiseBoundaryConstraint,
#     PointwiseInteriorConstraint,
# )
# from modulus.domain.inferencer import PointwiseInferencer
# from modulus.utils.io.plotter import InferencerPlotter
# from modulus.key import Key
# from modulus.eq.pdes.electromagnetic import PEC, SommerfeldBC, MaxwellFreqReal
# import numpy as np
# import matplotlib.pyplot as plt
#
# x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
#
#
# @modulus.main(config_path="conf", config_name="config")
# def run(cfg: ModulusConfig) -> None:
#     #  Define the components and their material properties
#
#     '''components = {
#         "glass1": {"box": Box((0, -1, -1), (5, 1, 1)), "epsilon": 0.375, "mu": 1},
#         "dielectric": {"box": Box((-0.5, -0.9, -0.5), (0.5, 0.9, 0.5)), "epsilon": 85, "mu": 1},
#         "glass2": {"box": Box((-5, -1, -1), (0, 1, 1)), "epsilon": 0.375, "mu": 1},
#         "air": {"box": Box((-0.5, -1, -1), (10, 1, 1)), "epsilon": 1, "mu": 1},
#     }
#
#     # Define the domain
#      domain = Domain(components=components)'''
#     components = {
#         "Glass1": {
#             "box": Box(x, 0, 5, y, -1, 1, z, -1, 1),
#             "permittivity": 0.375,
#             "permeability": 1,
#         },
#         "Dielectric": {
#             "box": Box(x, -0.5, 0.5, y, -0.9, 0.9, z, -0.5, 0.5),
#             "permittivity": 85,
#             "permeability": 1,
#         },
#         "Glass2": {
#             "box": Box(x, -5, 0, y, -1, 1, z, -1, 1),
#             "permittivity": 0.375,
#             "permeability": 1,
#         },
#         "Air": {
#             "box": Box(x, -0.5, 10, y, -1, 1, z, -1, 1),
#             "permittivity": 1,
#             "permeability": 1,
#         },
#     }
#
#     # Define the domain and add the components
#     domain = Domain()
#     for name, props in components.items():
#         domain.add_component(
#             name, props["box"], props["permittivity"], props["permeability"]
#         )
#         '''# Define the boundary and interior constraints
#         boundary_constraints = [
#             PointwiseBoundaryConstraint(
#                 Key("ElectricField", x, y, z, component=name, direction=dir),
#                 0,
#                 domain.get_boundary(name, dir),
#             )
#             for name in components.keys()
#             for dir in ["x_min", "x_max", "y_min", "y_max"]
#         ]
#         interior_constraints = [
#             PointwiseInteriorConstraint(
#                 Key("ElectricField", x, y, z, component=name, direction="z"),
#                 0,
#                 domain.get_interior(name),
#             )
#             for name in components.keys()
#         ]'''
#         # Define the boundary and interior constraints
#         '''boundary_constraints = [
#             PointwiseBoundaryConstraint(
#                 Key("ElectricField", x, y, z, direction=dir),
#                 0,
#                 domain.get_boundary(name, dir),
#             )
#             for name in components.keys()
#             for dir in ["x_min", "x_max", "y_min", "y_max"]
#         ]'''
#         Xmin = PointwiseBoundaryConstraint(Box((0, 0), (0, domain.width[1]), (0, domain.width[2])), PEC())
#         Xmax = PointwiseBoundaryConstraint(
#             Box((domain.width[0], domain.width[1]), (0, domain.width[1]), (0, domain.width[2])), PEC())
#         Ymin = PointwiseBoundaryConstraint(Box((0, domain.width[0]), (0, 0), (0, domain.width[2])), PEC())
#         Ymax = PointwiseBoundaryConstraint(
#             Box((0, domain.width[0]), (domain.width[1], domain.width[1]), (0, domain.width[2])), PEC())
#         Zmin = PointwiseBoundaryConstraint(Box((0, domain.width[0]), (0, domain.width[1]), (0, 0)), SommerfeldBC("z"))
#         Zmax = PointwiseBoundaryConstraint(
#             Box((0, domain.width[0]), (0, domain.width[1]), (domain.width[2], domain.width[2])), SommerfeldBC("z"))
#
#         domain.add_constraint(Xmin)
#         domain.add_constraint(Xmax)
#         domain.add_constraint(Ymin)
#         domain.add_constraint(Ymax)
#         domain.add_constraint(Zmin)
#         domain.add_constraint(Zmax)
#
#         interior_constraints = [
#             PointwiseInteriorConstraint(Key("ElectricField", x, y, z, component=name), 0, domain.get_interior(name), )
#             for name in components.keys()]
#
#         # Define the PDE system
#         '''pde_system = []
#         for name in components.keys():
#             pde_system.append(
#                 MaxwellFreqReal(
#                     Key("ElectricField", x, y, z, component=name),
#                     Key("MagneticField", x, y, z, component=name),
#                     domain.get_permittivity(name),
#                     domain.get_permeability(name),
#                 )
#             )
#             pde_system.append(
#                 PEC(
#                     Key("ElectricField", x, y, z, component=name),
#                     domain.get_boundary(name),
#                 )
#             )
#             pde_system.append(
#                 SommerfeldBC(
#                     Key("ElectricField", x, y, z, component=name),
#                     Key("MagneticField", x, y, z, component=name),
#                     domain.get_boundary(name),
#                 )
#             )'''
#         #  Define the components and their material properties
#
#         '''components = {
#             "glass1": {"box": Box((0, -1, -1), (5, 1, 1)), "epsilon": 0.375, "mu": 1},
#             "dielectric": {"box": Box((-0.5, -0.9, -0.5), (0.5, 0.9, 0.5)), "epsilon": 85, "mu": 1},
#             "glass2": {"box": Box((-5, -1, -1), (0, 1, 1)), "epsilon": 0.375, "mu": 1},
#             "air": {"box": Box((-0.5, -1, -1), (10, 1, 1)), "epsilon": 1, "mu": 1},
#         }
#
#         # Define the domain
#          domain = Domain(components=components)'''
#         '''components = {
#             "Glass1": {
#                 "box": Box(x, 0, 5, y, -1, 1, z, -1, 1),
#                 "permittivity": 0.375,
#                 "permeability": 1,
#             },
#             "Dielectric": {
#                 "box": Box(x, -0.5, 0.5, y, -0.9, 0.9, z, -0.5, 0.5),
#                 "permittivity": 85,
#                 "permeability": 1,
#             },
#             "Glass2": {
#                 "box": Box(x, -5, 0, y, -1, 1, z, -1, 1),
#                 "permittivity": 0.375,
#                 "permeability": 1,
#             },
#             "Air": {
#                 "box": Box(x, -0.5, 10, y, -1, 1, z, -1, 1),
#                 "permittivity": 1,
#                 "permeability": 1,
#             },
#         }
#
#         # Define the domain and add the components
#         domain = Domain()
#         for name, props in components.items():
#             domain.add_component(
#                 name, props["box"], props["permittivity"], props["permeability"]
#             )'''
#             '''# Define the boundary and interior constraints
#             boundary_constraints = [
#                 PointwiseBoundaryConstraint(
#                     Key("ElectricField", x, y, z, component=name, direction=dir),
#                     0,
#                     domain.get_boundary(name, dir),
#                 )
#                 for name in components.keys()
#                 for dir in ["x_min", "x_max", "y_min", "y_max"]
#             ]
#             interior_constraints = [
#                 PointwiseInteriorConstraint(
#                     Key("ElectricField", x, y, z, component=name, direction="z"),
#                     0,
#                     domain.get_interior(name),
#                 )
#                 for name in components.keys()
#             ]'''
#             # Define the boundary and interior constraints
#             boundary_constraints = [
#                 PointwiseBoundaryConstraint(
#                     Key("ElectricField", x, y, z, name, dir),
#                     0,
#                     domain.get_boundary(name, dir),
#                 )
#                 for name in components.keys()
#                 for dir in ["x_min", "x_max", "y_min", "y_max"]
#             ]
#             interior_constraints = [
#                 PointwiseInteriorConstraint(Key("ElectricField", x, y, z, name), 0,
#                                             domain.get_interior(name), )
#                 for name in components.keys()
#             ]
#
#             # Define the PDE system
#             '''pde_system = []
#             for name in components.keys():
#                 pde_system.append(
#                     MaxwellFreqReal(
#                         Key("ElectricField", x, y, z, component=name),
#                         Key("MagneticField", x, y, z, component=name),
#                         domain.get_permittivity(name),
#                         domain.get_permeability(name),
#                     )
#                 )
#                 pde_system.append(
#                     PEC(
#                         Key("ElectricField", x, y, z, component=name),
#                         domain.get_boundary(name),
#                     )
#                 )
#                 pde_system.append(
#                     SommerfeldBC(
#                         Key("ElectricField", x, y, z, component=name),
#                         Key("MagneticField", x, y, z, component=name),
#                         domain.get_boundary(name),
#                     )
#                 )'''
#         pde_system = []
#         for name in components.keys():
#             pde_system.append(
#                 MaxwellFreqReal(
#                     str(key("ElectricField", x, y, z, name)),
#                     str(key("MagneticField", x, y, z, name)),
#                     domain.get_permittivity(name),
#                     domain.get_permeability(name,)
#                 )
#             )
#             pde_system.append(
#                 PEC(
#                     str(key("ElectricField", x, y, z, name)),
#                     domain.get_boundary(name),
#
#                 )
#             )
#             pde_system.append(
#                 SommerfeldBC(
#                     str(key("ElectricField", x, y, z, name)),
#                     str(key("MagneticField", x, y, z, name)),
#                     domain.get_boundary(name),
#                 )
#             )
#
#
# # Add pointwise boundary constraints for PEC and waveguide port
# length = 10
# height = 5
# width = 5
# wall_PEC = PointwiseBoundaryConstraint(
#     nodes=nodes,
#     geometry=rec,
#     outvar={"PEC_x": 0.0, "PEC_y": 0.0, "PEC_z": 0.0},
#     batch_size=cfg.batch_size.PEC,
#     lambda_weighting={"PEC_x": 100.0, "PEC_y": 100.0, "PEC_z": 100.0},
#     criteria=And(~Eq(x, 0), ~Eq(x, width)),
# )
#
# waveguide_domain.add_constraint(wall_PEC, "PEC")
#
# waveguide_port = Number(0)
# for k in eigenmode:
#     waveguide_port += sin(k * pi * y / length) * sin(k * pi * z / height)
#
# Waveguide_port = PointwiseBoundaryConstraint(
#     nodes=nodes,
#     geometry=rec,
#     outvar={"uz": waveguide_port},
#     batch_size=cfg.batch_size.Waveguide_port,
#     lambda_weighting={"uz": 100.0},
#     criteria=Eq(x, 0),
# )
# waveguide_domain.add_constraint(Waveguide_port, "Waveguide_port")
#
#         # Define the boundary conditions
#         '''boundaries = {
#             "Xmin": {"constraint": PEC(), "box": Box((0, 0), None, None)},
#             "Xmax": {"constraint": PEC(), "box": Box((5, 5), None, None)},
#             "Ymin": {"constraint": PEC(), "box": Box(None, (-1, -1), None)},
#             "Ymax": {"constraint": PEC(), "box": Box(None, (1, 1), None)},
#             "Zmin": {"constraint": SommerfeldBC("z"), "box": Box(None, None, (-1, -1))},
#             "Zmax": {"constraint": SommerfeldBC("z"), "box": Box(None, None, (1, 1))},
#         }
#         for bname, b in boundaries.items():
#             domain.add_constraint(PointwiseBoundaryConstraint(b["box"], b["constraint"], name=bname))'''
#         boundaries = {
#             Xmin = PointwiseBoundaryConstraint (Box((0,0), None, None), PEC())
#             Xmin.name = "Xmin"
#             "Xmax": {"constraint": PEC(), "box": Box((5, 5), None, None)},
#             "Ymin": {"constraint": PEC(), "box": Box(None, (-1, -1), None)},
#             "Ymax": {"constraint": PEC(), "box": Box(None, (1, 1), None)},
#             "Zmin": {"constraint": SommerfeldBC("z"), "box": Box(None, None, (-1, -1))},
#             "Zmax": {"constraint": SommerfeldBC("z"), "box": Box(None, None, (1, 1))},
#         }
#
#
#         # Define the interior constraint (open)
#         domain.add_constraint(PointwiseInteriorConstraint(Box((-0.5, 5), (-1, 1), (-1, 1))))
#
#         # Define the inferencer
#         inference_keys = [
#             Key("efield_x", x, y, z),
#             Key("efield_y", x, y, z),
#             Key("efield_z", x, y, z),
#             Key("hfield_x", x, y, z),
#             Key("hfield_y", x, y, z),
#             Key("hfield_z", x, y, z),
#         ]
#         inference_keys += [
#             Key("intensity", x, y, z, component=c) for c in components.keys()
#         ]
#         inferencer = PointwiseInferencer(domain, inference_keys)
#
#         # Define the solver
#         freq_range = np.linspace(10e12, 20e12, num=101)
#
#         s11_vals = []
#         s21_vals = []
#
#         for freq in freq_range:
#             solver = Solver(
#                 freq, domain=domain, inferencer=inferencer,
#                 pde=MaxwellFreqReal(),
#                 config=ModulusConfig(solver="direct", use_analytical_jacobian=False),
#             )
#             solution = solver.solve()
#             s_params = solution.s_params("Xmin", "Xmax")
#             s11_vals.append(s_params[0][0])
#             s21_vals.append(s_params[1][0])
#
#             fig, ax = plt.subplots()
#             ax.plot(freq_range, np.abs(s11_vals), label="S11")
#             ax.plot(freq_range, np.abs(s21_vals), label="S21")
#             ax.set_xlabel("Frequency (Hz)")
#             ax.set_ylabel("Magnitude")
#             ax.legend()
#             # plt.show()
#             plt.savefig()
#
#     # Define the boundary conditions
#     boundaries = {
#         "Xmin": {"constraint": PEC(), "box": Box((0, 0), None, None)},
#         "Xmax": {"constraint": PEC(), "box": Box((5, 5), None, None)},
#         "Ymin": {"constraint": PEC(), "box": Box(None, (-1, -1), None)},
#         "Ymax": {"constraint": PEC(), "box": Box(None, (1, 1), None)},
#         "Zmin": {"constraint": SommerfeldBC("z"), "box": Box(None, None, (-1, -1))},
#         "Zmax": {"constraint": SommerfeldBC("z"), "box": Box(None, None, (1, 1))},
#     }
#     for bname, b in boundaries.items():
#         domain.add_constraint(PointwiseBoundaryConstraint(b["box"], b["constraint"], name=bname))
#
#     # Define the interior constraint (open)
#     domain.add_constraint(PointwiseInteriorConstraint(Box((-0.5, 5), (-1, 1), (-1, 1))))
#
#     # Define the inferencer
#     inference_keys = [
#         Key("efield_x", x, y, z),
#         Key("efield_y", x, y, z),
#         Key("efield_z", x, y, z),
#         Key("hfield_x", x, y, z),
#         Key("hfield_y", x, y, z),
#         Key("hfield_z", x, y, z),
#     ]
#     inference_keys += [
#         Key("intensity", x, y, z, component=c) for c in components.keys()
#     ]
#     inferencer = PointwiseInferencer(domain, inference_keys)
#
#     # Define the solver
#     freq_range = np.linspace(10e12, 20e12, num=101)
#
#     s11_vals = []
#     s21_vals = []
#
#     for freq in freq_range:
#         solver = Solver(
#             freq, domain=domain, inferencer=inferencer,
#             pde=MaxwellFreqReal(),
#             config=ModulusConfig(solver="direct", use_analytical_jacobian=False),
#         )
#         solution = solver.solve()
#         s_params = solution.s_params("Xmin", "Xmax")
#         s11_vals.append(s_params[0][0])
#         s21_vals.append(s_params[1][0])
#
#         fig, ax = plt.subplots()
#         ax.plot(freq_range, np.abs(s11_vals), label="S11")
#         ax.plot(freq_range, np.abs(s21_vals), label="S21")
#         ax.set_xlabel("Frequency (Hz)")
#         ax.set_ylabel("Magnitude")
#         ax.legend()
#         # plt.show()
#         plt.savefig()
#
#
# if __name__ == "__main__":
#     run()

from sympy import Symbol, pi, sin, Number, Eq
from sympy.logic.boolalg import Or

import modulus
from modulus.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.utils.io import csv_to_dict
from modulus.solver import Solver
from modulus.domain import Domain
from modulus.geometry.primitives_2d import Rectangle
from modulus.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.domain.validator import PointwiseValidator
from modulus.domain.inferencer import PointwiseInferencer
from modulus.utils.io.plotter import ValidatorPlotter, InferencerPlotter
from modulus.key import Key
from modulus.eq.pdes.wave_equation import HelmholtzEquation
from modulus.eq.pdes.navier_stokes import GradNormal
#from modulus.eq.boundary_conditions import BlochBoundaryCondition, RadiantBoundaryCondition


x, y, z = Symbol("x"), Symbol("y"), Symbol("z")


@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
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


