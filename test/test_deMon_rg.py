import os
import shutil
from copy import deepcopy

import numpy as np
import pytest
from ase.atoms import Atoms
from conftest import compute_numgrad

import deMonPy
from deMonPy.deMonNano import deMonNano
from deMonPy.molden import read_XYZ

deMonPy.configure_from_file("global.json")

parameters = {
    "DEMON_EXECUTABLE": deMonPy.DEMON_EXECUTABLE,
    "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
    "DEMON_PARAMETERS": {
        "ACTIVE": {
            "DFTB": {"SCC": True},
            "CHARGE": 0.0,
        },
    },
}

image = Atoms(
    [
        "C",
        "C",
        "N",
        "C",
        "C",
        "N",
        "C",
        "C",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "Ar",
    ],
    positions=np.array(
        [
            [1.188367820025961, -0.791963696891490, -0.110257297286295],
            [0.760376820025962, -0.020468696891491, 1.213427702713705],
            [-0.116809179974038, 1.033744303108510, 0.816564702713705],
            [-1.312762179974039, 0.526830303108510, 0.224073702713705],
            [-0.884737179974039, -0.244625696891490, -1.099629297286295],
            [0.535658820025961, -0.142239696891490, -1.201197297286295],
            [0.964475820025962, 1.212058303108510, -1.350483297286295],
            [0.536476820025961, 1.983416303108510, -0.027019297286295],
            [2.293157820025963, -0.757119696891491, -0.239642297286295],
            [0.868695820025961, -1.853187696891491, -0.046486297286295],
            [0.238654820025961, -0.717377696891490, 1.902302702713705],
            [1.656376820025962, 0.390792303108509, 1.729724702713705],
            [-1.820598179974038, -0.173774696891491, 0.919554702713705],
            [-2.010428179974038, 1.358783303108510, -0.020114297286295],
            [-1.190464179974038, -1.309529696891490, -1.029229297286295],
            [-1.373583179974038, 0.211010303108510, -1.989537297286295],
            [2.067746820025961, 1.256872303108509, -1.484343297286295],
            [0.483872820025962, 1.675070303108510, -2.240302297286294],
            [-0.152207179974038, 2.821315303108510, -0.273552297286295],
            [1.431709820025961, 2.403197303108510, 0.482362702713705],
            [-1.274156179974038, -2.711962696891490, 1.170045702713705],
        ]
    ),
)

amonia = Atoms(
    ["N", "H", "H", "H"],
    positions=np.array(
        [
            [3.909756207549047, -0.000000059186190, -2.347416764065182],
            [3.909756207549047, 0.939730940813810, -2.735713764065181],
            [4.723587207549047, -0.469865059186190, -2.735713764065181],
            [3.095925207549047, -0.469865059186190, -2.735713764065181],
        ]
    ),
)

test_force = np.array(
    [
        [-0.006963965319, 0.001254789683, -0.037576627105],
        [-0.022454509872, 0.029174135056, 0.010333308948],
        [0.011524956515, -0.022078400594, -0.036363221917],
        [0.016435640233, 0.018912236158, 0.028889962938],
        [0.031921113781, -0.009015766455, -0.019022634283],
        [-0.012186141464, 0.020666802885, 0.036969302993],
        [-0.001748590270, -0.034356119483, -0.015340060965],
        [-0.017330660062, -0.006269476598, 0.032849514255],
        [0.001349313292, 0.001350722211, 0.001355395932],
        [-0.004115413125, -0.001192393348, 0.003280469581],
        [-0.002367015775, -0.004359121696, -0.002146410617],
        [0.002113145845, -0.000026035185, -0.001007572168],
        [0.001085114839, -0.005271319954, -0.000495809252],
        [-0.000414203116, 0.000638984118, -0.002214104720],
        [-0.000669797043, -0.002097760183, 0.004923895189],
        [-0.001177624068, 0.002017795280, 0.000149504435],
        [0.001243747085, 0.000706180999, 0.001621555568],
        [-0.001578325918, 0.001452595124, 0.000273765565],
        [-0.000849082998, 0.000136559065, -0.001984468983],
        [0.001972887047, -0.000610493814, -0.000637861646],
        [0.004209410391, 0.008966086730, -0.003857903750],
    ]
)

WORKDIR = ".run/dftbrg/"


class TestDFTB_Rg:
    def test_argon(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                    "DISP": 2,
                    "POLA": True,
                    "NOPOLQM": True,
                },
                "RG": {"COUPLING": "ARGON", "ALPHARG": 11.07, "FILENAME": None},
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results

        assert np.allclose(results["energy"]["energy"], -19.51939925, atol=1e-7)

    def test_argon_opt(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                    "DISP": 2,
                    "POLA": True,
                    "NOPOLQM": True,
                },
                "RG": {"COUPLING": "ARGON", "ALPHARG": 11.07, "FILENAME": None},
            }
        )
        parameter_config.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"MAX": 999, "OUT": 1, "TRAJECTORY": True},
                    },
                }
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results

        assert np.allclose(results["energy"]["energy"], -19.53597981, atol=1e-7)

    def test_argon_md(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                    "DISP": 2,
                    "POLA": True,
                    "NOPOLQM": True,
                },
                "RG": {"COUPLING": "ARGON", "ALPHARG": 11.07, "FILENAME": None},
            }
        )
        parameter_config.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "MD": {
                            "MDYNAMICS": {
                                "ZERO": False,
                                "RANDOM": 155,
                            },
                            "TIMESTEP": 0.4,
                            "MDSTEP": {"MAX": 100, "OUT": 1, "SOUT": 1, "TSIM": None},
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

        traj = results["trajectory"]
        temperature = [atm.get_temperature() for atm in traj]

        assert temperature[0] < 155.1 and temperature[0] > 154.9

    def test_argon_cluster(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                    "DISP": 2,
                    "POLA": True,
                    "NOPOLQM": True,
                },
                "RG": {"COUPLING": "ARGON", "ALPHARG": 11.07, "FILENAME": None},
            }
        )
        images, ref = read_XYZ("test/data_test/rg_test.mol")

        for i, image in enumerate(images):
            dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
            dem.calculate(symbols=image.symbols, positions=image.positions)

            results = dem.results
            reference = float(ref[i].split()[3])

            assert np.allclose(results["energy"]["energy"], reference, atol=1e-7)

    def test_argon_cluster_ion(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                    "DISP": 2,
                    "POLA": True,
                    "NOPOLQM": True,
                },
                "CHARGE": 1.0,
                "RG": {"COUPLING": "ARGON", "ALPHARG": 11.07, "FILENAME": None},
            }
        )
        images, ref = read_XYZ("test/data_test/rg+_test.mol")

        for i, image in enumerate(images):
            dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
            dem.calculate(symbols=image.symbols, positions=image.positions)

            results = dem.results
            reference = float(ref[i].split()[3])

            assert np.allclose(results["energy"]["energy"], reference, atol=1e-7)

    def test_argon_pola(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                    "DISP": 2,
                    "POLA": True,
                    "NOPOLQM": True,
                },
                "RG": {
                    "COUPLING": "ARGON",
                    "ALPHARG": 1.107,
                },
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        assert np.allclose(results["energy"]["energy"], -19.51939793, atol=1e-7)

    def test_argon_ldep(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                    "DISP": 2,
                    "POLA": True,
                    "NOPOLQM": True,
                    "L-DEP": True,
                },
                "RG": {
                    "COUPLING": "ARGON",
                    "ALPHARG": 11.07,
                },
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        assert np.allclose(results["energy"]["energy"], -19.56988169, atol=1e-7)

    def test_argon_fermi(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                    "DISP": 2,
                    "POLA": True,
                    "NOPOLQM": True,
                    "FERMI": 50.00,
                },
                "RG": {
                    "COUPLING": "ARGON",
                    "ALPHARG": 11.07,
                },
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        assert np.allclose(results["energy"]["energy"], -19.51939925, atol=1e-7)

    @pytest.mark.beta
    def test_argon_dftb3(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                    "DISP": 2,
                    "POLA": True,
                    "NOPOLQM": True,
                    "THIRD": True,
                    "GCOR": 4.0
                },
                "RG": {
                    "COUPLING": "ARGON",
                    "ALPHARG": 11.07,
                },
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        shutil.copy2("test/data_test/3ord_param", f"{WORKDIR}/3ord_param")
        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        assert np.allclose(results["energy"]["energy"], -19.52046221, atol=1e-7)

    @pytest.mark.beta
    def test_argon_ci(self):

        _image = image + amonia

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                    "DISP": 2,
                    "POLA": True,
                    "NOPOLQM": True,
                },
                "CHARGE": 1.0,
                "CI": {
                    "CONST": 1,
                },
                "CUTSYS": {
                    "FRAGMENT": [20, 3],
                },
                "RG": {
                    "COUPLING": "ARGON",
                    "ALPHARG": 11.07,
                },
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=_image.symbols, positions=_image.positions)

        results = dem.results
        energy = results["energy"]
        assert np.allclose(energy["energy"], -22.75794379, atol=1e-7)

    @pytest.mark.beta
    def test_argon_const(self):

        _image = image + amonia

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                    "DISP": 2,
                    "POLA": True,
                    "NOPOLQM": True,
                },
                "CHARGE": 1.0,
                "CI": {"SIZECI": 2, "NOSLAT": False},
                "CUTSYS": {
                    "FRAGMENT": [20, 3],
                },
                "RG": {
                    "COUPLING": "ARGON",
                    "ALPHARG": 11.07,
                },
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=_image.symbols, positions=_image.positions)

        results = dem.results
        conf_1 = results["configuration_1"]
        conf_2 = results["configuration_2"]
        assert np.allclose(conf_1["energy"], -22.75794379, atol=1e-7)
        assert np.allclose(conf_2["energy"], -22.55212638, atol=1e-7)

    @pytest.mark.beta
    def test_argon_ci_exci(self):

        _image = image + amonia

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                    "DISP": 2,
                    "POLA": True,
                    "NOPOLQM": True,
                },
                "CHARGE": 1.0,
                "CI": {"SIZECI": 2, "NOSLAT": False, "EXCCI": 4},
                "CUTSYS": {
                    "FRAGMENT": [20, 3],
                },
                "RG": {
                    "COUPLING": "ARGON",
                    "ALPHARG": 11.07,
                },
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=_image.symbols, positions=_image.positions)

        results = dem.results
        conf_1 = results["configuration_1"]
        conf_2 = results["configuration_2"]

        assert np.allclose(conf_1["energy"], -22.75794379, atol=1e-7)
        assert np.allclose(conf_2["energy"], -22.66060063, atol=1e-7)

        assert np.allclose(len(results["states"]), 10.0)

        assert np.allclose(results["energy"]["energy"], -22.7625355, atol=1e-7)

    def test_argon_freq(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                    "DISP": 2,
                    "POLA": True,
                    "NOPOLQM": True,
                },
                "FREQ": True,
                "RG": {
                    "COUPLING": "ARGON",
                    "ALPHARG": 11.07,
                },
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        assert np.allclose(results["zpe"], 0.18155212, atol=1e-4)
        assert len(results["frequency"]) == 63
        mode_1 = results["frequency"][34]
        assert mode_1["mode"] == 35
        assert np.allclose(mode_1["frequency"], 1402.3, atol=1e-1)
        assert np.allclose(mode_1["intensity"], 132.0, atol=1e-1)


    def test_argon_tddftb(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                    "DISP": 2,
                    "POLA": True,
                    "NOPOLQM": True,
                },
                "TD-DFTB": True,
                "RG": {
                    "COUPLING": "ARGON",
                    "ALPHARG": 11.07,
                },
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]
        assert energy["energy"] == -19.51939925

        assert "triplet" in results.keys()
        assert "singlet" in results.keys()



    @pytest.mark.xfail(reason="EXPERIMENTAL")
    def test_argon_read(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                    "DISP": 2,
                    "POLA": True,
                    "NOPOLQM": True,
                },
                "RG": {"COUPLING": "READ", "ALPHARG": 11.07},
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        shutil.copy2("./data_test/rg_parameters", f"{WORKDIR}/rg_parameters")

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results

        assert np.allclose(results["energy"]["energy"], -19.51939925, atol=1e-7)

    @pytest.mark.forces
    def test_argon_grad(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                    "DISP": 2,
                    "POLA": True,
                    "NOPOLQM": True,
                },
                "RG": {
                    "COUPLING": "ARGON",
                    "ALPHARG": 11.07,
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

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config_bis)
        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results

        assert np.allclose(results["energy"]["energy"], -19.51939925, atol=1e-7)

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
        grad = compute_numgrad(
            symbols=image.symbols, positions=image.positions, calculator=dem, delta=0.001
        )

        assert np.allclose(results["forces"], grad, atol=1e-5)

    def test_argon_wmull(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                    "DISP": 2,
                    "POLA": True,
                    "NOPOLQM": True,
                },
                "RG": {
                    "COUPLING": "ARGON",
                    "ALPHARG": 11.07,
                },
                "WMULL": {
                    "BONDPARAMS": {
                        "C H": 0.08,
                        "N H": 0.38,
                        "N C": 0.4,
                    },
                },
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -19.47889372, atol=1e-7)

    def test_argon_cm3(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                    "DISP": 2,
                    "POLA": True,
                    "NOPOLQM": True,
                },
                "RG": {
                    "COUPLING": "ARGON",
                    "ALPHARG": 11.07,
                },
                "CM3": {
                    "BONDPARAMS": {
                        "C H": 0.08,
                        "N H": 0.38,
                        "N C": 0.4,
                    },
                },
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -19.28123631, atol=1e-7)

    @pytest.mark.xfail(reason="NOT CRITICAL -> TO FIX")
    @pytest.mark.forces
    def test_argon_wmull_grad(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                    "DISP": 2,
                    "POLA": True,
                    "NOPOLQM": True,
                },
                "WMULL": {
                    "BONDPARAMS": {
                        "C H": 0.08,
                        "N H": 0.38,
                        "N C": 0.4,
                    },
                },
                "RG": {
                    "COUPLING": "ARGON",
                    "ALPHARG": 11.07,
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

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config_bis)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -19.47889372, atol=1e-7)

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
        grad = compute_numgrad(
            symbols=image.symbols, positions=image.positions, calculator=dem, delta=0.001
        )

        assert np.allclose(results["forces"], grad, atol=1e-5)

    @pytest.mark.forces
    def test_argon_cm3_grad(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                    "DISP": 2,
                    "POLA": True,
                    "NOPOLQM": True,
                },
                "CM3": {
                    "BONDPARAMS": {
                        "C H": 0.08,
                        "N H": 0.38,
                        "N C": 0.4,
                    },
                },
                "RG": {
                    "COUPLING": "ARGON",
                    "ALPHARG": 11.07,
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

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config_bis)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -19.28123631, atol=1e-7)

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
        grad = compute_numgrad(
            symbols=image.symbols, positions=image.positions, calculator=dem, delta=0.001
        )

        assert np.allclose(results["forces"], grad, atol=1e-5)

    def test_argon_dipole(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                    "DISP": 2,
                    "POLA": True,
                    "NOPOLQM": True,
                },
                "RG": {
                    "COUPLING": "ARGON",
                    "ALPHARG": 11.07,
                },
                "DIPOLE": {
                    "OUTFILE": True,
                },
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        assert np.allclose(results["energy"]["energy"], -19.51939925, atol=1e-7)

        filename = os.path.join(dem.workdir, "deMon.dip")
        assert os.path.exists(filename)

        dipole = results["tensors"]["dipole_norm"]
        assert np.allclose(dipole, 0.10091401, 1e-5)

    def test_argon_mdyn(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                    "DISP": 2,
                    "POLA": True,
                    "NOPOLQM": True,
                },
                "RG": {
                    "COUPLING": "ARGON",
                    "ALPHARG": 11.07,
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

    def test_argon_mdyn_constraint(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                    "DISP": 2,
                    "POLA": True,
                    "NOPOLQM": True,
                },
                "RG": {
                    "COUPLING": "ARGON",
                    "ALPHARG": 11.07,
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

    def test_argon_ptmc(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                    "DISP": 2,
                    "POLA": True,
                    "NOPOLQM": True,
                },
                "RG": {
                    "COUPLING": "ARGON",
                    "ALPHARG": 11.07,
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

    
    def test_argon_ptmc_rigid(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                    "DISP": 2,
                    "POLA": True,
                    "NOPOLQM": True,
                },
                "RG": {
                    "COUPLING": "ARGON",
                    "ALPHARG": 11.07,
                },
                "CUTSYS": {"FRAGMENT": [20, 1], "RIGID": True},
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
