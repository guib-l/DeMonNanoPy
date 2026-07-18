import copy
import os

import numpy as np
from ase.atoms import Atoms

import deMonPy
from deMonPy.deMonNano import deMonNano

deMonPy.configure_from_file(os.path.join("..", "global.json"))

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

WORKDIR = ".run/mdbath/"


class TestMDbath:
    def test_md_noBath(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "MD": {
                            "MDYNAMICS": {
                                "ZERO": True,
                                "RANDOM": 200,
                            },
                            "TIMESTEP": 0.4,
                            "MDSTEP": {"MAX": 548, "OUT": 1, "SOUT": 1, "TSIM": None},
                            "TRAJECTORY": True,
                            "MDBATH": {
                                "SCAL": False,
                                "BERE": False,
                                "NOSE": False,
                                "LANGE": False,
                                "STOCH_R": False,
                                "ANDERSEN": False,
                                "LOCA": False,
                                "VAL": 0.5,
                                "NTHER": 4,
                                "FREQTH": 100,
                            },
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

        temperature = [atm.get_temperature() for atm in traj]
        diff = np.abs(temperature[0] - 200)
        assert diff < 0.1, "Starting temperature is not tacken into account"

    def test_md_scalling(self):
        # TODO: Vérifier le scaling factor

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "MD": {
                            "MDYNAMICS": {
                                "ZERO": True,
                                "RANDOM": 200,
                            },
                            "TIMESTEP": 0.4,
                            "MDSTEP": {"MAX": 548, "OUT": 1, "SOUT": 1, "TSIM": None},
                            "TRAJECTORY": True,
                            "MDBATH": {
                                "SCAL": True,
                                "BERE": False,
                                "NOSE": False,
                                "LANGE": False,
                                "STOCH_R": False,
                                "ANDERSEN": False,
                                "LOCA": False,
                                "VAL": 0.5,
                                "NTHER": 4,
                                "FREQTH": 100,
                            },
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

        temperature = [atm.get_temperature() for atm in traj]
        diff = np.abs(temperature[0] - 200)
        assert diff < 0.1, "Starting temperature is not tacken into account"

    def test_md_berendson(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "MD": {
                            "MDYNAMICS": {
                                "ZERO": True,
                                "RANDOM": 200,
                            },
                            "TIMESTEP": 0.4,
                            "MDSTEP": {"MAX": 548, "OUT": 1, "SOUT": 1, "TSIM": None},
                            "TRAJECTORY": True,
                            "MDBATH": {
                                "SCAL": False,
                                "BERE": True,
                                "NOSE": False,
                                "LANGE": False,
                                "STOCH_R": False,
                                "ANDERSEN": False,
                                "LOCA": False,
                                "VAL": 0.5,
                                "NTHER": 4,
                                "FREQTH": 100,
                            },
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

        temperature = [atm.get_temperature() for atm in traj]
        diff = np.abs(temperature[0] - 200)
        assert diff < 0.1, "Starting temperature is not tacken into account"

    def test_md_locberendson(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "MD": {
                            "MDYNAMICS": {
                                "ZERO": True,
                                "RANDOM": 200,
                            },
                            "TIMESTEP": 0.4,
                            "MDSTEP": {"MAX": 548, "OUT": 1, "SOUT": 1, "TSIM": None},
                            "TRAJECTORY": True,
                            "MDBATH": {
                                "SCAL": False,
                                "BERE": False,
                                "NOSE": False,
                                "LANGE": False,
                                "STOCH_R": False,
                                "ANDERSEN": False,
                                "LOCA": True,
                                "VAL": 0.5,
                                "NTHER": 4,
                                "FREQTH": 100,
                            },
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

        temperature = [atm.get_temperature() for atm in traj]
        diff = np.abs(temperature[0] - 200)
        assert diff < 0.1, "Starting temperature is not tacken into account"

    def test_md_anderson(self):
        # TODO: Fix correct test

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "MD": {
                            "MDYNAMICS": {
                                "ZERO": True,
                                "RANDOM": 200,
                            },
                            "TIMESTEP": 0.4,
                            "MDSTEP": {"MAX": 548, "OUT": 1, "SOUT": 1, "TSIM": None},
                            "TRAJECTORY": True,
                            "MDBATH": {
                                "SCAL": False,
                                "BERE": False,
                                "NOSE": False,
                                "LANGE": False,
                                "STOCH_R": False,
                                "ANDERSEN": True,
                                "LOCA": False,
                                "VAL": 0.5,
                                "NTHER": 4,
                                "FREQTH": 100,
                            },
                        },
                    }
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        results["potential_energy"]
        results["kinetic_energy"]
        results["total_energy"]

        traj = results["trajectory"]
        assert len(traj) == 549, "MAX criteria is not conserved"

        temperature = [atm.get_temperature() for atm in traj]
        diff = np.abs(temperature[0] - 200)
        assert diff < 0.1, "Starting temperature is not tacken into account"

    def test_md_langevin(self):
        # TODO: Fix correct test

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "MD": {
                            "MDYNAMICS": {
                                "ZERO": True,
                                "RANDOM": 200,
                            },
                            "TIMESTEP": 0.4,
                            "MDSTEP": {"MAX": 548, "OUT": 1, "SOUT": 1, "TSIM": None},
                            "TRAJECTORY": True,
                            "MDBATH": {
                                "SCAL": False,
                                "BERE": False,
                                "NOSE": False,
                                "LANGE": True,
                                "STOCH_R": False,
                                "ANDERSEN": False,
                                "LOCA": False,
                                "VAL": 0.5,
                                "NTHER": 4,
                                "FREQTH": 100,
                            },
                        },
                    }
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        results["potential_energy"]
        results["kinetic_energy"]
        results["total_energy"]

        traj = results["trajectory"]
        assert len(traj) == 549, "MAX criteria is not conserved"

        temperature = [atm.get_temperature() for atm in traj]
        diff = np.abs(temperature[0] - 200)
        assert diff < 0.1, "Starting temperature is not tacken into account"

    def test_md_stochr(self):
        # TODO: Fix correct test

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "MD": {
                            "MDYNAMICS": {
                                "ZERO": True,
                                "RANDOM": 200,
                            },
                            "TIMESTEP": 0.4,
                            "MDSTEP": {"MAX": 548, "OUT": 1, "SOUT": 1, "TSIM": None},
                            "TRAJECTORY": True,
                            "MDBATH": {
                                "SCAL": False,
                                "BERE": False,
                                "NOSE": False,
                                "LANGE": False,
                                "STOCH_R": True,
                                "ANDERSEN": False,
                                "LOCA": False,
                                "VAL": 0.5,
                                "NTHER": 4,
                                "FREQTH": 100,
                            },
                        },
                    }
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        results["potential_energy"]
        results["kinetic_energy"]
        results["total_energy"]

        traj = results["trajectory"]
        assert len(traj) == 549, "MAX criteria is not conserved"

        temperature = [atm.get_temperature() for atm in traj]
        diff = np.abs(temperature[0] - 200)
        assert diff < 0.1, "Starting temperature is not tacken into account"

    def test_md_nose(self):
        # TODO: Fix correct test

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "MD": {
                            "MDYNAMICS": {
                                "ZERO": True,
                                "RANDOM": 200,
                            },
                            "TIMESTEP": 0.4,
                            "MDSTEP": {"MAX": 548, "OUT": 1, "SOUT": 1, "TSIM": None},
                            "TRAJECTORY": True,
                            "MDBATH": {
                                "SCAL": False,
                                "BERE": False,
                                "NOSE": True,
                                "LANGE": False,
                                "STOCH_R": False,
                                "ANDERSEN": False,
                                "LOCA": False,
                                "VAL": 0.5,
                                "NTHER": 4,
                                "FREQTH": 100,
                            },
                        },
                    }
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        results["potential_energy"]
        results["kinetic_energy"]
        results["total_energy"]

        traj = results["trajectory"]
        assert len(traj) == 549, "MAX criteria is not conserved"

        temperature = [atm.get_temperature() for atm in traj]
        diff = np.abs(temperature[0] - 200)
        assert diff < 0.1, "Starting temperature is not tacken into account"

    def test_md_nose10(self):
        # TODO: Fix correct test

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "MD": {
                            "MDYNAMICS": {
                                "ZERO": True,
                                "RANDOM": 200,
                            },
                            "TIMESTEP": 0.4,
                            "MDSTEP": {"MAX": 548, "OUT": 1, "SOUT": 1, "TSIM": None},
                            "TRAJECTORY": True,
                            "MDBATH": {
                                "SCAL": False,
                                "BERE": False,
                                "NOSE": True,
                                "LANGE": False,
                                "STOCH_R": False,
                                "ANDERSEN": False,
                                "LOCA": False,
                                "VAL": 0.5,
                                "NTHER": 10,
                                "FREQTH": 100,
                            },
                        },
                    }
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        results["potential_energy"]
        results["kinetic_energy"]
        results["total_energy"]

        traj = results["trajectory"]
        assert len(traj) == 549, "MAX criteria is not conserved"

        temperature = [atm.get_temperature() for atm in traj]
        diff = np.abs(temperature[0] - 200)
        assert diff < 0.1, "Starting temperature is not tacken into account"
