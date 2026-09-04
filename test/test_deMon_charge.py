from copy import deepcopy

import numpy as np
import pytest
from ase.atoms import Atoms
from conftest import compute_numgrad

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


WORKDIR_ion = ".run/dftbion/"
WORKDIR_csy = ".run/dftbcutsys/"
WORKDIR_mlt = ".run/dftbmulti/"


class TestCharges:
    def test_charged(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CHARGE": 1.0,
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR_ion, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -7.6911342
        assert energy["electronic_energy"] == -7.84792412
        assert energy["coulomb_energy"] == 0.15950647
        assert energy["repulsive_energy"] == 0.15678992

    def test_wmull(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "WMULL": {
                    "BONDPARAMS": {
                        "O H": 0.18,
                    },
                }
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR_ion, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -8.04442048
        assert energy["electronic_energy"] == -8.2012104
        assert energy["coulomb_energy"] == 0.05453568
        assert energy["repulsive_energy"] == 0.15678992

    @pytest.mark.forces
    def test_wmull_grad(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "WMULL": {
                    "BONDPARAMS": {
                        "O H": 0.18,
                    },
                }
            }
        )
        parameter_config_bis = deepcopy(parameter_config)
        parameter_config_bis.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"SP": True, "TRAJECTORY": True},
                    },
                }
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR_ion, **parameter_config_bis)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -8.04442048
        assert energy["electronic_energy"] == -8.2012104
        assert energy["coulomb_energy"] == 0.05453568
        assert energy["repulsive_energy"] == 0.15678992

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR_ion, **parameter_config)
        grad = compute_numgrad(
            symbols=image.symbols, positions=image.positions, calculator=dem, delta=0.001
        )

        assert np.allclose(results["forces"], grad, atol=1e-5)

    def test_cm3pot(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CM3": {
                    "BONDPARAMS": {
                        "O H": 0.08,
                    },
                }
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR_ion, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -8.0368677
        assert energy["electronic_energy"] == -8.19365762
        assert energy["coulomb_energy"] == 0.06029717

    @pytest.mark.forces
    def test_cm3pot_grad(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CM3": {
                    "BONDPARAMS": {
                        "O H": 0.08,
                    },
                }
            }
        )
        parameter_config_bis = deepcopy(parameter_config)
        parameter_config_bis.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"SP": True, "TRAJECTORY": True},
                    },
                }
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR_ion, **parameter_config_bis)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -8.0368677
        assert energy["electronic_energy"] == -8.19365762
        assert energy["coulomb_energy"] == 0.06029717
        assert energy["repulsive_energy"] == 0.15678992

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR_ion, **parameter_config)
        grad = compute_numgrad(
            symbols=image.symbols, positions=image.positions, calculator=dem, delta=0.001
        )

        assert np.allclose(results["forces"], grad, atol=1e-5)

    def test_cm3inter(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CM3INTER": {
                    "BONDPARAMS": {
                        "O H": 0.08,
                    },
                }
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR_ion, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -8.06209343

    @pytest.mark.forces
    def test_cm3inter_grad(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CM3INTER": {
                    "BONDPARAMS": {
                        "O H": 0.08,
                    },
                }
            }
        )
        parameter_config_bis = deepcopy(parameter_config)
        parameter_config_bis.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"SP": True, "TRAJECTORY": True},
                    },
                }
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR_ion, **parameter_config_bis)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -8.06209343

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR_ion, **parameter_config)
        grad = compute_numgrad(
            symbols=image.symbols, positions=image.positions, calculator=dem, delta=0.001
        )

        assert np.allclose(results["forces"], grad, atol=1e-5)


class TestCutSys:
    def test_cutsys(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CUTSYS": {"FRAGMENT": [3, 3], "RIGID": False},
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR_csy, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -8.06209343
        assert energy["electronic_energy"] == -8.21888334
        assert energy["coulomb_energy"] == 0.04185358
        assert energy["repulsive_energy"] == 0.15678992

    def test_cutsys_opt(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CUTSYS": {"FRAGMENT": [3, 3], "RIGID": False},
            }
        )
        parameter_config.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"MAX": 99, "TRAJECTORY": True},
                    },
                }
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR_csy, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -8.15543481

    def test_cutsys_opt_rigid(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CUTSYS": {"FRAGMENT": [3, 3], "RIGID": True},
            }
        )
        parameter_config.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"MAX": 99, "TRAJECTORY": True},
                    },
                }
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR_csy, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -8.11990885

        ref = np.linalg.norm(
            results["trajectory"][0].positions[:3] - results["trajectory"][0].positions[0]
        )
        probe = np.linalg.norm(
            results["trajectory"][-1].positions[:3] - results["trajectory"][-1].positions[0]
        )

        assert np.allclose(
            ref,
            probe,
            atol=1e-5,
        )

    def test_cutsys_rigid(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CUTSYS": {"FRAGMENT": [3, 3], "RIGID": True},
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR_csy, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -8.06209343
        assert energy["electronic_energy"] == -8.21888334
        assert energy["coulomb_energy"] == 0.04185358
        assert energy["repulsive_energy"] == 0.15678992

    def test_cutsys_natmol(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CUTSYS": {"FRAGMENT": [], "NMOL": 2, "NATMOL": 3},
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR_csy, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -8.06209343
        assert energy["electronic_energy"] == -8.21888334
        assert energy["coulomb_energy"] == 0.04185358
        assert energy["repulsive_energy"] == 0.15678992

    def test_cutsys_opt_natmol(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CUTSYS": {"FRAGMENT": [], "NMOL": 2, "NATMOL": 3},
            }
        )
        parameter_config.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"MAX": 99, "TRAJECTORY": True},
                    },
                }
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR_csy, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -8.15543481

    @pytest.mark.xfail(reason="NOT CRITICAL -> TO FIX")
    def test_cutsys_ptmc(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CUTSYS": {"FRAGMENT": [3, 3], "RIGID": False},
            }
        )
        parameter_config.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "PTMC": {
                            "MC": {"MAX": 10, "SEED": 3141592, "WALL": None},
                            "MCTEMP": {
                                "NTEMP": 12,
                                "LINEAR": True,
                                "TEMPMIN": 30,
                                "TEMPMAX": 300,
                            },
                        },
                    }
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR_csy, **parameter_config)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        ptmc = results["ptmc"]

        assert len(ptmc["temperatures"]) == 12
        assert ptmc["seeds"][0] == 3141592
        assert ptmc["nb_step"] == 10
        assert ptmc["optout"] == 1
        assert ptmc["nb_temp"] == 12
        assert ptmc["exchange"]["method"] == "SE"
        assert ptmc["exchange"]["start_after"] == 100
        assert ptmc["exchange"]["each_step"] == 10
        assert ptmc["exchange"]["swap_probability"] == 100

    @pytest.mark.xfail(reason="NOT CRITICAL -> TO FIX")
    def test_cutsys_ptmc_rigid(self):

        image = Atoms(
            ["O", "H", "H", "C", "O", "O"],
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

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CUTSYS": {"FRAGMENT": [3, 3], "RIGID": True},
            }
        )
        parameter_config.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "PTMC": {
                            "MC": {"MAX": 10, "SEED": 3141592, "WALL": None},
                            "MCTEMP": {
                                "NTEMP": 12,
                                "GEOM": True,
                                "TEMPMIN": 30,
                                "TEMPMAX": 300,
                            },
                        },
                    }
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR_csy, **parameter_config)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        ptmc = results["ptmc"]

        assert len(ptmc["temperatures"]) == 12
        assert ptmc["seeds"][0] == 3141592
        assert ptmc["nb_step"] == 10
        assert ptmc["optout"] == 1
        assert ptmc["nb_temp"] == 12
        assert ptmc["exchange"]["method"] == "SE"
        assert ptmc["exchange"]["start_after"] == 100
        assert ptmc["exchange"]["each_step"] == 10
        assert ptmc["exchange"]["swap_probability"] == 100


class TestMultipl:
    def test_multiplicity_1(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "MULTI": 1,
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR_mlt, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -8.06209343
        assert energy["electronic_energy"] == -8.21888334
        assert energy["coulomb_energy"] == 0.04185358
        assert energy["repulsive_energy"] == 0.15678992

    def test_multiplicity_3(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "MULTI": 3,
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR_mlt, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -7.69430622
        assert energy["electronic_energy"] == -7.85109614
        assert energy["coulomb_energy"] == 0.03543676
        assert energy["repulsive_energy"] == 0.15678992

    @pytest.mark.forces
    def test_multiplicity_3_grad(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "MULTI": 3,
            }
        )
        parameter_config_bis = deepcopy(parameter_config)
        parameter_config_bis.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"SP": True, "TRAJECTORY": True},
                    },
                }
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR_mlt, **parameter_config_bis)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -7.69430622

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR_mlt, **parameter_config)
        grad = compute_numgrad(
            symbols=image.symbols, positions=image.positions, calculator=dem, delta=0.001
        )

        assert np.allclose(results["forces"], grad, atol=1e-5)

    def test_multiplicity_3_cm3(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "MULTI": 3,
                "CM3": {
                    "BONDPARAMS": {
                        "O H": 0.08,
                    },
                },
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR_mlt, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -7.68089083

    @pytest.mark.forces
    def test_multiplicity_3_cm3_grad(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "MULTI": 3,
                "CM3": {
                    "BONDPARAMS": {
                        "O H": 0.08,
                    },
                },
            }
        )
        parameter_config_bis = deepcopy(parameter_config)
        parameter_config_bis.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"SP": True, "TRAJECTORY": True},
                    },
                }
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR_mlt, **parameter_config_bis)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -7.68089083

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR_mlt, **parameter_config)
        grad = compute_numgrad(
            symbols=image.symbols, positions=image.positions, calculator=dem, delta=0.001
        )

        assert np.allclose(results["forces"], grad, atol=1e-5)

    def test_multiplicity_3_wmull(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "MULTI": 3,
                "WMULL": {
                    "BONDPARAMS": {
                        "O H": 0.08,
                    },
                },
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR_mlt, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -7.68983001

    @pytest.mark.forces
    def test_multiplicity_3_wmull_grad(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "MULTI": 3,
                "WMULL": {
                    "BONDPARAMS": {
                        "O H": 0.08,
                    },
                },
            }
        )
        parameter_config_bis = deepcopy(parameter_config)
        parameter_config_bis.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"SP": True, "TRAJECTORY": True},
                    },
                }
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR_mlt, **parameter_config_bis)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -7.68983001

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR_mlt, **parameter_config)
        grad = compute_numgrad(
            symbols=image.symbols, positions=image.positions, calculator=dem, delta=0.001
        )

        assert np.allclose(results["forces"], grad, atol=1e-5)
