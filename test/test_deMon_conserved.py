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

WORKDIR = ".run/mdcsvd/"


class TestMDconserved:
    def test_md_conserve_com(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "MD": {
                            "MDYNAMICS": {
                                "RANDOM": 200,
                            },
                            "CONSERVE": {"ALL": False, "COM": True, "ANG": False, "MOM": False},
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

        coms = np.array([img.get_center_of_mass() for img in traj])
        assert max(coms[:, 0]) < 1e-5, ""
        assert max(coms[:, 1]) < 1e-5, ""
        assert max(coms[:, 2]) < 1e-5, ""

    def test_md_conserve_ang(self):
        # TODO: Implement test

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "MD": {
                            "MDYNAMICS": {
                                "RANDOM": 200,
                            },
                            "CONSERVE": {"ALL": False, "COM": False, "ANG": True, "MOM": False},
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

    def test_md_conserve_mom(self):
        # TODO: Implement test

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "MD": {
                            "MDYNAMICS": {
                                "RANDOM": 200,
                            },
                            "CONSERVE": {"ALL": False, "COM": False, "ANG": False, "MOM": True},
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

    def test_md_conserve_all(self):
        # TODO: Implement tests

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "MD": {
                            "MDYNAMICS": {
                                "RANDOM": 200,
                            },
                            "CONSERVE": {"ALL": True, "COM": False, "ANG": False, "MOM": False},
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
