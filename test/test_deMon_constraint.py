import copy

import numpy as np
from ase.atoms import Atoms

import deMonPy
from deMonPy.deMonNano import deMonNano

deMonPy.configure_from_file("global.json")
parameters = {
    "DEMON_EXECUTABLE": deMonPy.DEMON_EXECUTABLE,
    "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
    "DEMON_PARAMETERS": {
        "ACTIVE": {
            "DFTB": {"SCC": True},
        },
    },
}

image = Atoms(
    ["O", "H", "H", "O", "H", "H"],
    positions=np.array(
        [
            [1.2478, -0.5185, 3.4049],
            [1.5946, -1.4204, 3.3886],
            [0.9008, -0.3341, 2.5062],
            [3.2478, -0.4185, 3.4049],
            [3.5946, -1.5204, 3.3886],
            [2.9008, -0.3341, 2.6062],
        ]
    ),
)

WORKDIR = ".run/mdconst/"


class TestMDconstraints:
    def test_md_constraint_positions(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "MD": {
                            "MDYNAMICS": {
                                "RANDOM": 200,
                            },
                            "MDCONSTRAINTS": {
                                "POSITION": "1-3",
                            },
                            "TIMESTEP": 0.4,
                            "MDSTEP": {
                                "MAX": 548,
                                "OUT": 1,
                                "SOUT": 1,
                            },
                            "TRAJECTORY": True,
                        },
                    }
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        pote = results["potential_energy"]
        kine = results["kinetic_energy"]
        tote = results["total_energy"]

        diff = np.sum((tote - (pote + kine))[1:])
        assert diff <= 1e-5, "Energy conserved"

        traj = results["trajectory"]
        assert len(traj) == 549, "MAX criteria is not conserved"

    def test_md_constraint_positions_2(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "MD": {
                            "MDYNAMICS": {
                                "RANDOM": 200,
                            },
                            "MDCONSTRAINTS": {
                                "POSITION": "1,2,3",
                            },
                            "TIMESTEP": 0.4,
                            "MDSTEP": {
                                "MAX": 548,
                                "OUT": 1,
                                "SOUT": 1,
                            },
                            "TRAJECTORY": True,
                        },
                    }
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        pote = results["potential_energy"]
        kine = results["kinetic_energy"]
        tote = results["total_energy"]

        diff = np.sum((tote - (pote + kine))[1:])
        assert diff <= 1e-5, "Energy conserved"

        traj = results["trajectory"]
        assert len(traj) == 549, "MAX criteria is not conserved"

    def test_md_constraint_X(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "MD": {
                            "MDYNAMICS": {
                                "RANDOM": 200,
                            },
                            "MDCONSTRAINTS": {
                                "X": "1,2,3",
                            },
                            "TIMESTEP": 0.4,
                            "MDSTEP": {
                                "MAX": 548,
                                "OUT": 1,
                                "SOUT": 1,
                            },
                            "TRAJECTORY": True,
                        },
                    }
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        pote = results["potential_energy"]
        kine = results["kinetic_energy"]
        tote = results["total_energy"]

        diff = np.sum((tote - (pote + kine))[1:])
        assert diff <= 1e-5, "Energy conserved"

        traj = results["trajectory"]
        assert len(traj) == 549, "MAX criteria is not conserved"
