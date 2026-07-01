import os
import shutil
from copy import deepcopy

import numpy as np
import pytest
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
    @pytest.mark.bird
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
        dem = deMonNano(
            title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config
        )

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results

        assert np.allclose(results["energy"]["energy"], -19.51939925, atol=1e-7)

    @pytest.mark.bird
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
        dem = deMonNano(
            title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config
        )

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        print(results["energy"]["energy"])
        assert np.allclose(results["energy"]["energy"], -19.51939793, atol=1e-7)

    @pytest.mark.forces
    @pytest.mark.bird
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
                "RG": {"COUPLING": "ARGON", "ALPHARG": 11.07, "FILENAME": None},
            }
        )
        parameter_config.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {
                            "SP": True,
                        },
                    },
                }
            }
        )

        dem = deMonNano(
            title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config
        )

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results

        assert np.allclose(results["energy"]["energy"], -19.51939925, atol=1e-7)
        assert np.allclose(results["forces"], test_force, atol=1e-7)

    @pytest.mark.bird
    @pytest.mark.xfail(reason="NOT CRITICAL -> TO FIX")
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
        dem = deMonNano(
            title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config
        )

        shutil.copy2("./data_test/rg_parameters", f"{WORKDIR}/rg_parameters")

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results

        assert np.allclose(results["energy"]["energy"], -19.51939925, atol=1e-7)
