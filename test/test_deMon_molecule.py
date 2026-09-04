import copy

# import configs
import numpy as np
import pytest
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
            "QUATERNION": {"BOHR": False, "ANGST": False, "COORDS": None, "RIGID": False},
            "MOLECULES": {
                "NAMES": [],
            },
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

WORKDIR = ".run/molecule/"


class TestMolecule:
    def test_molecule(self):
        import shutil

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "MOLECULES": {"NAMES": ["WAT", "BZZ", "WAT"]},
                "QUATERNION": {
                    "COORDS": np.array(
                        [
                            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, -3.0, 0.0, 0.7, 0.7, 0.0, 0.0],
                            [1.0, -3.0, -4.0, 1.0, 0.0, 0.0, 0.0],
                        ]
                    )
                },
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        shutil.copy2("test/data_test/MOLECULES", f"{WORKDIR}/MOLECULES")

        mod.calculate(symbols=image.symbols, positions=image.positions, clean_repository=False)

        results = mod.results
        assert np.allclose(results["energy"]["energy"], -20.7065151, atol=1e-7)

    def test_molecule_angst(self):
        import shutil

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "MOLECULES": {"NAMES": ["WAT", "BZZ", "WAT"]},
                "QUATERNION": {
                    "ANGST": True,
                    "COORDS": np.array(
                        [
                            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, -3.0, 0.0, 0.7, 0.7, 0.0, 0.0],
                            [1.0, -3.0, -4.0, 1.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                },
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        shutil.copy2("test/data_test/MOLECULES", f"{WORKDIR}/MOLECULES")

        mod.calculate(symbols=image.symbols, positions=image.positions, clean_repository=False)

        results = mod.results
        assert np.allclose(results["energy"]["energy"], -20.7065151, atol=1e-7)

    def test_molecule_bohr(self):
        import shutil

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "MOLECULES": {"NAMES": ["WAT", "BZZ", "WAT"]},
                "QUATERNION": {
                    "BOHR": True,
                    "COORDS": np.array(
                        [
                            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, -3.0, 0.0, 0.7, 0.7, 0.0, 0.0],
                            [1.0, -3.0, -4.0, 1.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                },
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        shutil.copy2("test/data_test/MOLECULES", f"{WORKDIR}/MOLECULES")

        mod.calculate(symbols=image.symbols, positions=image.positions, clean_repository=False)

        results = mod.results
        assert np.allclose(results["energy"]["energy"], -1.06599675, atol=1e-7)

    @pytest.mark.beta
    def test_molecule_rigid(self):
        import shutil

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "MOLECULES": {"NAMES": ["WAT", "BZZ", "WAT"]},
                "QUATERNION": {
                    "RIGID": True,
                    "COORDS": np.array(
                        [
                            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, -3.0, 0.0, 0.7, 0.7, 0.0, 0.0],
                            [1.0, -3.0, -4.0, 1.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                },
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        shutil.copy2("test/data_test/MOLECULES", f"{WORKDIR}/MOLECULES")

        mod.calculate(symbols=image.symbols, positions=image.positions, clean_repository=False)

        results = mod.results
        assert np.allclose(results["energy"]["energy"], -20.7065151, atol=1e-7)

    def test_molecule_opt(self):
        import shutil

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "MOLECULES": {"NAMES": ["WAT", "BZZ", "WAT"]},
                "QUATERNION": {
                    "COORDS": np.array(
                        [
                            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, -3.0, 0.0, 0.7, 0.7, 0.0, 0.0],
                            [1.0, -3.0, -4.0, 1.0, 0.0, 0.0, 0.0],
                        ]
                    )
                },
            }
        )
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"MAX": 999, "OUT": 1, "TRAJECTORY": True},
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        shutil.copy2("test/data_test/MOLECULES", f"{WORKDIR}/MOLECULES")

        mod.calculate(symbols=image.symbols, positions=image.positions, clean_repository=False)

        results = mod.results
        assert np.allclose(results["energy"]["energy"], -20.7269581, atol=1e-7)

    @pytest.mark.beta
    def test_molecule_opt_rigid(self):
        import shutil

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "MOLECULES": {"NAMES": ["WAT", "BZZ", "WAT"]},
                "QUATERNION": {
                    "RIGID": True,
                    "COORDS": np.array(
                        [
                            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, -3.0, 0.0, 0.7, 0.7, 0.0, 0.0],
                            [1.0, -3.0, -4.0, 1.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                },
            }
        )
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"MAX": 999, "OUT": 1, "TRAJECTORY": True},
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        shutil.copy2("test/data_test/MOLECULES", f"{WORKDIR}/MOLECULES")

        mod.calculate(symbols=image.symbols, positions=image.positions, clean_repository=False)

        results = mod.results
        assert np.allclose(results["energy"]["energy"], -20.7269581, atol=1e-7)

    @pytest.mark.beta
    def test_molecule_freq(self):
        import shutil

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "FREQ": True,
                "MOLECULES": {"NAMES": ["WAT", "BZZ", "WAT"]},
                "QUATERNION": {
                    "COORDS": np.array(
                        [
                            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, -3.0, 0.0, 0.7, 0.7, 0.0, 0.0],
                            [1.0, -3.0, -4.0, 1.0, 0.0, 0.0, 0.0],
                        ]
                    )
                },
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        shutil.copy2("test/data_test/MOLECULES", f"{WORKDIR}/MOLECULES")

        mod.calculate(symbols=image.symbols, positions=image.positions, clean_repository=False)

        results = mod.results
        assert np.allclose(results["zpe"], 0.14335175, atol=1e-4)
        assert len(results["frequency"]) == 54
        mode_1 = results["frequency"][0]
        assert mode_1["mode"] == 1
        assert np.allclose(mode_1["frequency"], -587.7, atol=1e-1)
        assert np.allclose(mode_1["intensity"], 9.5, atol=1e-1)

    @pytest.mark.xfail(reason="NOT CRITICAL -> TO FIX")
    @pytest.mark.beta
    def test_molecule_ci(self):
        import shutil

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True},
                "CI": {"SIZECI": 3},
                "CUTSYS": {
                    "RIGID": False,
                    "FRAGMENT": [3, 12, 3],
                },
                "MOLECULES": {"NAMES": ["WAT", "BZZ", "WAT"]},
                "QUATERNION": {
                    "COORDS": np.array(
                        [
                            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, -3.0, 0.0, 0.7, 0.7, 0.0, 0.0],
                            [1.0, -3.0, -4.0, 1.0, 0.0, 0.0, 0.0],
                        ]
                    )
                },
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        shutil.copy2("test/data_test/MOLECULES", f"{WORKDIR}/MOLECULES")

        mod.calculate(symbols=image.symbols, positions=image.positions, clean_repository=False)

        results = mod.results
        assert np.allclose(results["energy"]["energy"], -20.7065151, atol=1e-7)

    @pytest.mark.beta
    def test_molecule_const(self):
        import shutil

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True},
                "CI": {"CONST": 1},
                "CUTSYS": {
                    "RIGID": False,
                    "FRAGMENT": [3, 12, 3],
                },
                "MOLECULES": {"NAMES": ["WAT", "BZZ", "WAT"]},
                "QUATERNION": {
                    "COORDS": np.array(
                        [
                            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, -3.0, 0.0, 0.7, 0.7, 0.0, 0.0],
                            [1.0, -3.0, -4.0, 1.0, 0.0, 0.0, 0.0],
                        ]
                    )
                },
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        shutil.copy2("test/data_test/MOLECULES", f"{WORKDIR}/MOLECULES")

        mod.calculate(symbols=image.symbols, positions=image.positions, clean_repository=False)

        results = mod.results
        assert np.allclose(results["energy"]["energy"], -20.70651510, atol=1e-7)

    @pytest.mark.beta
    def test_molecule_cutsys(self):
        import shutil

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True},
                "CUTSYS": {
                    "RIGID": False,
                    "FRAGMENT": [3, 12, 3],
                },
                "MOLECULES": {"NAMES": ["WAT", "BZZ", "WAT"]},
                "QUATERNION": {
                    "COORDS": np.array(
                        [
                            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, -3.0, 0.0, 0.7, 0.7, 0.0, 0.0],
                            [1.0, -3.0, -4.0, 1.0, 0.0, 0.0, 0.0],
                        ]
                    )
                },
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        shutil.copy2("test/data_test/MOLECULES", f"{WORKDIR}/MOLECULES")

        mod.calculate(symbols=image.symbols, positions=image.positions, clean_repository=False)

        results = mod.results
        assert np.allclose(results["energy"]["energy"], -20.70651510, atol=1e-7)

    @pytest.mark.beta
    def test_molecule_mdyn(self):

        parameter_config = copy.deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                },
                "MOLECULES": {"NAMES": ["WAT", "BZZ", "WAT"]},
                "QUATERNION": {
                    "COORDS": np.array(
                        [
                            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, -3.0, 0.0, 0.7, 0.7, 0.0, 0.0],
                            [1.0, -3.0, -4.0, 1.0, 0.0, 0.0, 0.0],
                        ]
                    )
                },
            }
        )
        parameter_config.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "MD": {
                            "MDYNAMICS": {
                                "ZERO": True,
                                "RANDOM": 300,
                            },
                            "TIMESTEP": 0.4,
                            "MDSTEP": {"MAX": 150, "OUT": 1, "SOUT": 1, "TSIM": None},
                            "MDTEMP": 300,
                            "TRAJECTORY": True,
                        },
                    },
                }
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        pote = results["potential_energy"]
        kine = results["kinetic_energy"]
        tote = results["total_energy"]

        assert np.sum((tote - (pote + kine))[1:]) <= 1e-5

    @pytest.mark.beta
    def test_molecule_mdyn_constraint(self):

        parameter_config = copy.deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                },
                "MOLECULES": {"NAMES": ["WAT", "BZZ", "WAT"]},
                "QUATERNION": {
                    "COORDS": np.array(
                        [
                            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, -3.0, 0.0, 0.7, 0.7, 0.0, 0.0],
                            [1.0, -3.0, -4.0, 1.0, 0.0, 0.0, 0.0],
                        ]
                    )
                },
            }
        )
        parameter_config.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "MD": {
                            "MDYNAMICS": {
                                "ZERO": True,
                                "RANDOM": 300,
                            },
                            "MDCONSTRAINTS": {
                                "POSITION": "1-3",
                            },
                            "TIMESTEP": 0.4,
                            "MDSTEP": {"MAX": 150, "OUT": 1, "SOUT": 1, "TSIM": None},
                            "MDTEMP": 300,
                            "TRAJECTORY": True,
                        },
                    },
                }
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        pote = results["potential_energy"]
        kine = results["kinetic_energy"]
        tote = results["total_energy"]

        assert np.sum((tote - (pote + kine))[1:]) <= 1e-5
        assert np.allclose(
            results["trajectory"][0].positions[:3],
            results["trajectory"][-1].positions[:3],
            atol=1e-5,
        )

    @pytest.mark.beta
    def test_molecule_ptmc(self):

        parameter_config = copy.deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                },
                "MOLECULES": {"NAMES": ["WAT", "BZZ", "WAT"]},
                "QUATERNION": {
                    "COORDS": np.array(
                        [
                            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, -3.0, 0.0, 0.7, 0.7, 0.0, 0.0],
                            [1.0, -3.0, -4.0, 1.0, 0.0, 0.0, 0.0],
                        ]
                    )
                },
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

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        ptmc = results["ptmc"]

        assert len(ptmc["temperatures"]) == 12
        assert ptmc["seeds"][0] == 3141592
        assert ptmc["nb_step"] == 10
        assert ptmc["optout"] == 1
        assert ptmc["nb_temp"] == 12
        assert ptmc["exchange"]["method"] == "SE"
        assert ptmc["exchange"]["start_after"] == 100
        assert ptmc["exchange"]["each_step"] == 10
        assert ptmc["exchange"]["swap_probability"] == 100.0
