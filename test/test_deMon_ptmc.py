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

WORKDIR = ".run/ptmc/"


class TestPTMC:
    def test_ptmc(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "PTMC": {
                            "MC": {"MAX": 150, "SEED": 3141592, "WALL": None},
                            "MCTEMP": {
                                "TMC": 300,
                                "NTEMP": 12,
                                "GEOM": True,
                                "LINEAR": False,
                                "TEMPMIN": 30,
                                "TEMPMAX": 300,
                                "X": None,
                                "Q": None,
                                "RESCALE": 10,
                                "SWAP": "AE",
                                "SMOD": 10,
                                "SPERCENT": 0.1,
                                "SDBG": True,
                                "OUT": 1,
                                "SOUT": 1,
                            },
                        },
                    }
                }
            }
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters
        )

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        ptmc = results["ptmc"]

        assert len(ptmc["temperatures"]) == 12
        assert ptmc["seeds"][0] == 3141592
        assert ptmc["nb_step"] == 150
        assert ptmc["optout"] == 1
        assert ptmc["nb_temp"] == 12
        assert ptmc["exchange"]["method"] == "All-exchanges parallel tempering"
        assert ptmc["exchange"]["start_after"] == 100
        assert ptmc["exchange"]["each_step"] == 10
        assert ptmc["exchange"]["swap_probability"] == 0.1

    def test_ptmc_exchangeNO(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "PTMC": {
                            "MC": {"MAX": 150, "SEED": 3141592, "WALL": None},
                            "MCTEMP": {
                                "TMC": 300,
                                "NTEMP": 12,
                                "GEOM": True,
                                "LINEAR": False,
                                "TEMPMIN": 30,
                                "TEMPMAX": 300,
                                "X": None,
                                "Q": None,
                                "RESCALE": 10,
                                "SWAP": "NO",
                                "SMOD": 10,
                                "SPERCENT": 0.1,
                                "SDBG": True,
                                "OUT": 1,
                                "SOUT": 1,
                            },
                        },
                    }
                }
            }
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters
        )

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        ptmc = results["ptmc"]
        energy = results["energy"]

        assert energy["energy"] == -8.12997879

        assert len(ptmc["temperatures"]) == 12
        assert ptmc["seeds"][0] == 3141592
        assert ptmc["nb_step"] == 150
        assert ptmc["optout"] == 1
        assert ptmc["nb_temp"] == 12
        assert ptmc["exchange"]["method"] == "None"
        assert ptmc["exchange"]["start_after"] == 100
        assert ptmc["exchange"]["each_step"] == 10
        assert ptmc["exchange"]["swap_probability"] == 0.1

    def test_ptmc_exchangeSE(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "PTMC": {
                            "MC": {"MAX": 150, "SEED": 3141592, "WALL": None},
                            "MCTEMP": {
                                "TMC": 300,
                                "NTEMP": 12,
                                "GEOM": True,
                                "LINEAR": False,
                                "TEMPMIN": 30,
                                "TEMPMAX": 300,
                                "X": None,
                                "Q": None,
                                "RESCALE": 10,
                                "SWAP": "SE",
                                "SMOD": 10,
                                "SPERCENT": 0.1,
                                "SDBG": True,
                                "OUT": 1,
                                "SOUT": 1,
                            },
                        },
                    }
                }
            }
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters
        )

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        ptmc = results["ptmc"]
        energy = results["energy"]

        assert energy["energy"] == -8.13308321

        assert len(ptmc["temperatures"]) == 12
        assert ptmc["seeds"][0] == 3141592
        assert ptmc["nb_step"] == 150
        assert ptmc["optout"] == 1
        assert ptmc["nb_temp"] == 12
        assert ptmc["exchange"]["method"] == "SE"
        assert ptmc["exchange"]["start_after"] == 100
        assert ptmc["exchange"]["each_step"] == 10
        assert ptmc["exchange"]["swap_probability"] == 0.1

    def test_ptmc_linear(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "PTMC": {
                            "MC": {"MAX": 150, "SEED": 3141592, "WALL": None},
                            "MCTEMP": {
                                "TMC": 300,
                                "NTEMP": 12,
                                "GEOM": False,
                                "LINEAR": True,
                                "TEMPMIN": 30,
                                "TEMPMAX": 300,
                                "X": None,
                                "Q": None,
                                "RESCALE": 10,
                                "SWAP": "AE",
                                "SMOD": 10,
                                "SPERCENT": 0.1,
                                "SDBG": True,
                                "OUT": 1,
                                "SOUT": 1,
                            },
                        },
                    }
                }
            }
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters
        )

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        ptmc = results["ptmc"]

        assert len(ptmc["temperatures"]) == 12
        temp = np.array([t["temperature"] for t in ptmc["temperatures"]])
        diff = np.sum(
            temp
            - np.array(
                [
                    30.00,
                    54.55,
                    79.09,
                    103.64,
                    128.18,
                    152.73,
                    177.27,
                    201.82,
                    226.36,
                    250.91,
                    275.45,
                    300.00,
                ]
            )
        )
        assert diff < 1e-5
        assert ptmc["seeds"][0] == 3141592
        assert ptmc["nb_step"] == 150
        assert ptmc["optout"] == 1
        assert ptmc["nb_temp"] == 12
        assert ptmc["exchange"]["method"] == "All-exchanges parallel tempering"
        assert ptmc["exchange"]["start_after"] == 100
        assert ptmc["exchange"]["each_step"] == 10
        assert ptmc["exchange"]["swap_probability"] == 0.1

    def test_mc(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "PTMC": {
                            "MC": {"MAX": 150, "SEED": 3141592, "WALL": None},
                            "MCTEMP": {"TMC": 300, "OUT": 1, "SOUT": 1},
                        },
                    }
                }
            }
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters
        )

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        ptmc = results["ptmc"]

        assert ptmc["seeds"][0] == 3141592
        assert ptmc["nb_step"] == 150
        assert ptmc["optout"] == 1
        assert ptmc["nb_temp"] == 1
        assert ptmc["exchange"]["each_step"] == 10

    def test_ptmc_initx(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "PTMC": {
                            "MC": {"MAX": 150, "SEED": 3141592, "WALL": None},
                            "MCTEMP": {
                                "NTEMP": 12,
                                "GEOM": True,
                                "LINEAR": False,
                                "TEMPMIN": 30,
                                "TEMPMAX": 300,
                                "X": 0.2,
                                "Q": None,
                                "RESCALE": 10,
                                "SWAP": "AE",
                                "SMOD": 10,
                                "SPERCENT": 0.1,
                                "SDBG": True,
                                "OUT": 1,
                                "SOUT": 1,
                            },
                        },
                    }
                }
            }
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters
        )

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        ptmc = results["ptmc"]

        assert len(ptmc["temperatures"]) == 12
        assert ptmc["seeds"][0] == 3141592
        assert ptmc["nb_step"] == 150
        assert ptmc["optout"] == 1
        assert ptmc["nb_temp"] == 12
        assert ptmc["exchange"]["method"] == "All-exchanges parallel tempering"
        assert ptmc["exchange"]["start_after"] == 100
        assert ptmc["exchange"]["each_step"] == 10
        assert ptmc["exchange"]["swap_probability"] == 0.1

    def test_ptmc_initq(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "PTMC": {
                            "MC": {"MAX": 150, "SEED": 3141592, "WALL": None},
                            "MCTEMP": {
                                "NTEMP": 12,
                                "GEOM": True,
                                "LINEAR": False,
                                "TEMPMIN": 30,
                                "TEMPMAX": 300,
                                "X": None,
                                "Q": 0.2,
                                "RESCALE": 10,
                                "SWAP": "AE",
                                "SMOD": 10,
                                "SPERCENT": 0.1,
                                "SDBG": True,
                                "OUT": 1,
                                "SOUT": 1,
                            },
                        },
                    }
                }
            }
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters
        )

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        ptmc = results["ptmc"]

        assert len(ptmc["temperatures"]) == 12
        assert ptmc["seeds"][0] == 3141592
        assert ptmc["nb_step"] == 150
        assert ptmc["optout"] == 1
        assert ptmc["nb_temp"] == 12
        assert ptmc["exchange"]["method"] == "All-exchanges parallel tempering"
        assert ptmc["exchange"]["start_after"] == 100
        assert ptmc["exchange"]["each_step"] == 10
        assert ptmc["exchange"]["swap_probability"] == 0.1

    def test_ptmc_rescale(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "PTMC": {
                            "MC": {"MAX": 150, "SEED": 3141592, "WALL": None},
                            "MCTEMP": {
                                "NTEMP": 12,
                                "GEOM": True,
                                "LINEAR": False,
                                "TEMPMIN": 30,
                                "TEMPMAX": 300,
                                "X": None,
                                "Q": None,
                                "RESCALE": 100,
                                "SWAP": "AE",
                                "SMOD": 10,
                                "SPERCENT": 0.1,
                                "SDBG": True,
                                "OUT": 1,
                                "SOUT": 1,
                            },
                        },
                    }
                }
            }
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters
        )

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        ptmc = results["ptmc"]

        assert len(ptmc["temperatures"]) == 12
        assert ptmc["seeds"][0] == 3141592
        assert ptmc["nb_step"] == 150
        assert ptmc["optout"] == 1
        assert ptmc["nb_temp"] == 12
        assert ptmc["exchange"]["method"] == "All-exchanges parallel tempering"
        assert ptmc["exchange"]["start_after"] == 100
        assert ptmc["exchange"]["each_step"] == 10
        assert ptmc["exchange"]["swap_probability"] == 0.1

    def test_ptmc_spercent(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "PTMC": {
                            "MC": {"MAX": 150, "SEED": 3141592, "WALL": None},
                            "MCTEMP": {
                                "NTEMP": 12,
                                "GEOM": True,
                                "LINEAR": False,
                                "TEMPMIN": 30,
                                "TEMPMAX": 300,
                                "X": None,
                                "Q": None,
                                "RESCALE": 10,
                                "SWAP": "AE",
                                "SMOD": 10,
                                "SPERCENT": 0.2,
                                "SDBG": True,
                                "OUT": 1,
                                "SOUT": 1,
                            },
                        },
                    }
                }
            }
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters
        )

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        ptmc = results["ptmc"]

        assert len(ptmc["temperatures"]) == 12
        assert ptmc["seeds"][0] == 3141592
        assert ptmc["nb_step"] == 150
        assert ptmc["optout"] == 1
        assert ptmc["nb_temp"] == 12
        assert ptmc["exchange"]["method"] == "All-exchanges parallel tempering"
        assert ptmc["exchange"]["start_after"] == 100
        assert ptmc["exchange"]["each_step"] == 10
        assert ptmc["exchange"]["swap_probability"] == 0.2

    def test_ptmc_out(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "PTMC": {
                            "MC": {"MAX": 150, "SEED": 3141592, "WALL": None},
                            "MCTEMP": {
                                "NTEMP": 12,
                                "GEOM": True,
                                "LINEAR": False,
                                "TEMPMIN": 30,
                                "TEMPMAX": 300,
                                "X": None,
                                "Q": None,
                                "RESCALE": 10,
                                "SWAP": "AE",
                                "SMOD": 10,
                                "SPERCENT": 0.1,
                                "SDBG": True,
                                "OUT": 10,
                                "SOUT": 1,
                            },
                            "TRAJECTORY": True,
                        },
                    }
                }
            }
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters
        )

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        ptmc = results["ptmc"]

        assert len(ptmc["temperatures"]) == 12
        assert ptmc["seeds"][0] == 3141592
        assert ptmc["nb_step"] == 150
        assert ptmc["optout"] == 1
        assert ptmc["nb_temp"] == 12
        assert ptmc["exchange"]["method"] == "All-exchanges parallel tempering"
        assert ptmc["exchange"]["start_after"] == 100
        assert ptmc["exchange"]["each_step"] == 10
        assert ptmc["exchange"]["swap_probability"] == 0.1
