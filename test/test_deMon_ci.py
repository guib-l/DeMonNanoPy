import os
import shutil
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
            "CHARGE": 1.0,
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
        "O",
        "H",
        "H",
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
            [-1.715265179974038, -2.828611696891490, 2.023954702713705],
            [-1.295279179974038, -3.579516696891490, 0.740793702713705],
        ]
    ),
)

water = Atoms(
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

WORKDIR = ".run/dftbci/"


class TestDFTBCI:
    def test_ci(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CI": {
                    "SIZECI": 2,
                },
                "CUTSYS": {
                    "FRAGMENT": [20, 3],
                },
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        conf_1 = results["configuration_1"]
        conf_2 = results["configuration_2"]

        assert np.allclose(conf_1["energy"], -23.34473263, atol=1e-7)
        assert np.allclose(conf_2["energy"], -23.10000944, atol=1e-7)

    @pytest.mark.forces
    def test_ci_grad(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CI": {
                    "SIZECI": 2,
                },
                "CUTSYS": {
                    "FRAGMENT": [20, 3],
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
        conf_1 = results["configuration_1"]
        conf_2 = results["configuration_2"]

        assert np.allclose(conf_1["energy"], -23.34473263, atol=1e-7)
        assert np.allclose(conf_2["energy"], -23.10000944, atol=1e-7)
        assert np.allclose(results["energy"]["energy"], -23.3449257719042, atol=1e-7)

        _gradient = np.array(
            [
                [-0.008596062054, 0.009517421669, -0.002829737275],
                [-0.006135784507, 0.005076993335, -0.010437115527],
                [-0.000926298607, 0.001546048898, 0.002621871908],
                [0.013157986670, 0.000063826038, -0.001269807598],
                [0.010697159984, 0.004493046851, 0.006328773669],
                [0.000788079736, -0.001535814335, -0.002673232944],
                [-0.006155783349, -0.007855916034, 0.008563118158],
                [-0.003477912306, -0.012677877494, 0.000289924375],
                [0.004230149522, -0.000554331081, -0.001140410750],
                [-0.000776527285, -0.003114928308, -0.000349738306],
                [-0.001630899850, -0.001567220959, 0.002296499767],
                [0.002962422073, 0.001730979814, 0.002779901037],
                [-0.001476884102, -0.001736232987, 0.002425156234],
                [-0.002937094928, 0.003300651394, -0.000051581876],
                [-0.000630535868, -0.003263764248, -0.000192221710],
                [-0.001672341368, 0.001022905505, -0.003960389384],
                [0.003933437693, -0.000199022458, -0.001339251312],
                [-0.001227695050, 0.001165723497, -0.003798537937],
                [-0.002447740307, 0.003362402133, -0.000029238779],
                [0.002710816600, 0.002002103774, 0.002436691286],
                [-0.000879491535, -0.002155580449, 0.000976880979],
                [0.000416187287, 0.000384538275, -0.000845936421],
                [0.000074811550, 0.000994047171, 0.000198382408],
            ]
        )

        assert np.allclose(results["forces"], _gradient, atol=1e-5)

    def test_const(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CI": {"CONST": 1},
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -23.28746978, atol=1e-7)
        assert np.allclose(energy["electronic_energy"], -23.92276564, atol=1e-7)
        assert np.allclose(energy["coulomb_energy"], 0.12999541, atol=1e-7)
        assert np.allclose(energy["repulsive_energy"], 0.63529586, atol=1e-7)

    # @pytest.mark.forces
    def test_const_grad(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CI": {
                    "CONST": 1,
                },
                "CUTSYS": {
                    "FRAGMENT": [20, 3],
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

        assert np.allclose(energy["energy"], -23.34492577, atol=1e-7)

        _gradient = np.array(
            [
                [-0.008548332282, 0.009677203810, -0.002865133512],
                [-0.006068552117, 0.005203602518, -0.010531591557],
                [-0.000850362251, 0.001529595276, 0.002628268351],
                [0.013155034948, 0.000133360973, -0.001359370672],
                [0.010672248802, 0.004600363747, 0.006305938213],
                [0.000851722133, -0.001529712308, -0.002628704366],
                [-0.006158452724, -0.007879425840, 0.008545338107],
                [-0.003494372224, -0.012676351958, 0.000314680124],
                [0.004229842414, -0.000544757678, -0.001139534668],
                [-0.000982071658, -0.003288887570, -0.000242962094],
                [-0.001805246901, -0.001800147305, 0.002303940882],
                [0.002965059192, 0.001735415143, 0.002771604208],
                [-0.001451863830, -0.001895163524, 0.002475398656],
                [-0.002938784005, 0.003292088976, -0.000046036162],
                [-0.000627987474, -0.003380387501, -0.000073309230],
                [-0.001673557238, 0.001013445433, -0.003956350855],
                [0.003931476051, -0.000196837067, -0.001333654328],
                [-0.001226834232, 0.001166399193, -0.003796978417],
                [-0.002446119065, 0.003361763783, -0.000029996998],
                [0.002712229523, 0.001998234078, 0.002431886842],
                [-0.000898614064, -0.001907083583, 0.000826045882],
                [0.000495905978, 0.000391586346, -0.000817146085],
                [0.000157631022, 0.000995695058, 0.000217667678],
            ]
        )

        assert np.allclose(results["forces"], _gradient, atol=1e-5)

    @pytest.mark.beta
    def test_ci_noslat(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CI": {"SIZECI": 2, "NOSLAT": True},
                "CUTSYS": {
                    "FRAGMENT": [20, 3],
                },
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        conf_1 = results["configuration_1"]
        conf_2 = results["configuration_2"]

        assert np.allclose(conf_1["energy"], -23.34473263, atol=1e-7)
        assert np.allclose(conf_2["energy"], -23.10000944, atol=1e-7)

    @pytest.mark.beta
    def test_ci_excci(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CI": {"SIZECI": 2, "NOSLAT": False, "EXCCI": 4},
                "CUTSYS": {
                    "FRAGMENT": [20, 3],
                },
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        conf_1 = results["configuration_1"]
        conf_2 = results["configuration_2"]

        assert np.allclose(conf_1["energy"], -23.34473263, atol=1e-7)
        assert np.allclose(conf_2["energy"], -23.24749332, atol=1e-7)

        assert np.allclose(len(results["states"]), 10.0)

    @pytest.mark.beta
    def test_const_opt(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CI": {"CONST": 1},
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
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

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.6781923, atol=1e-7)

    @pytest.mark.beta
    def test_ci_opt(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CI": {
                    "SIZECI": 2,
                },
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
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

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.71029648, atol=1e-7)

    @pytest.mark.beta
    def test_excci_opt(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CI": {
                    "SIZECI": 2,
                    "NOSLAT": False,
                    "EXCCI": 4,
                },
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
            }
        )
        parameter_config.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"MAX": 10, "OUT": 1, "TRAJECTORY": True},
                    },
                }
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.70495473, atol=1e-7)

    def test_ci_dftb3(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True, "THIRD": True, "GCOR": 4.0},
                "CI": {
                    "SIZECI": 2,
                },
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        shutil.copy2("test/data_test/3ord_param", f"{WORKDIR}/3ord_param")
        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.64664757, atol=1e-7)

    def test_const_dftb3(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True, "THIRD": True, "GCOR": 4.0},
                "CI": {
                    "CONST": 1,
                },
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        shutil.copy2("test/data_test/3ord_param", f"{WORKDIR}/3ord_param")
        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.57951117, atol=1e-7)

    def test_ci_fermi(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True, "FERMI": 50},
                "CI": {
                    "SIZECI": 2,
                    "NOSLAT": False,
                },
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.65119769, atol=1e-7)

    def test_const_fermi(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True, "FERMI": 50},
                "CI": {
                    "CONST": 1,
                },
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.58567456, atol=1e-7)

    def test_ci_ldep(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True, "L-DEP": True},
                "CI": {
                    "SIZECI": 2,
                    "NOSLAT": False,
                },
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.77607094, atol=1e-7)

    def test_const_ldep(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True, "L-DEP": True},
                "CI": {
                    "CONST": 1,
                },
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.5846698, atol=1e-7)

    @pytest.mark.xfail(reason="NOT CRITICAL -> TO FIX")
    @pytest.mark.forces
    def test_ci_ldep_grad(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True, "L-DEP": True},
                "CI": {
                    "SIZECI": 2,
                },
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
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

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.77607094, atol=1e-7)

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
        grad = compute_numgrad(
            symbols=water.symbols, positions=water.positions, calculator=dem, delta=0.001
        )

        assert np.allclose(results["forces"], grad, atol=1e-5)

    @pytest.mark.forces
    def test_ci_fermi_grad(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True, "FERMI": 50},
                "CI": {
                    "SIZECI": 2,
                },
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
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

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.65119769, atol=1e-7)

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
        grad = compute_numgrad(
            symbols=water.symbols, positions=water.positions, calculator=dem, delta=0.001
        )

        assert np.allclose(results["forces"], grad, atol=1e-5)

    def test_ci_disp1(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True, "DISP": 1},
                "CI": {"SIZECI": 2},
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.63790536, atol=1e-7)

    def test_const_disp1(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True, "DISP": 1},
                "CI": {"CONST": 1},
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.57238223, atol=1e-7)

    @pytest.mark.forces
    def test_ci_disp1_grad(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True, "DISP": 1},
                "CI": {"SIZECI": 2},
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
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

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.63790536, atol=1e-7)

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
        grad = compute_numgrad(
            symbols=water.symbols, positions=water.positions, calculator=dem, delta=0.001
        )

        assert np.allclose(results["forces"], grad, atol=1e-5)

    def test_ci_disp2(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True, "DISP": 2},
                "CI": {"SIZECI": 2},
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.65087386, atol=1e-7)

    def test_const_disp2(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True, "DISP": 2},
                "CI": {"CONST": 1},
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.58535073, atol=1e-7)

    @pytest.mark.forces
    def test_ci_disp2_grad(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True, "DISP": 2},
                "CI": {"SIZECI": 2},
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
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

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.65087386, atol=1e-7)

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
        grad = compute_numgrad(
            symbols=water.symbols, positions=water.positions, calculator=dem, delta=0.001
        )

        assert np.allclose(results["forces"], grad, atol=1e-5)

    def test_ci_freq(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                },
                "FREQ": True,
                "CI": {"SIZECI": 2},
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        assert np.allclose(results["zpe"], 0.0458589, atol=1e-4)
        assert len(results["frequency"]) == 18
        mode_1 = results["frequency"][0]
        assert mode_1["mode"] == 1
        assert np.allclose(mode_1["frequency"], -1323.4, atol=1e-1)
        assert np.allclose(mode_1["intensity"], 54.8, atol=1e-1)

    def test_const_freq(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                },
                "FREQ": True,
                "CI": {"CONST": 1},
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        assert np.allclose(results["zpe"], 0.04424336, atol=1e-4)
        assert len(results["frequency"]) == 18
        mode_1 = results["frequency"][0]
        assert mode_1["mode"] == 1
        assert np.allclose(mode_1["frequency"], -1490.6, atol=1e-1)
        assert np.allclose(mode_1["intensity"], 77.8, atol=1e-1)

    def test_ci_cm3(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True, "DISP": 2},
                "CM3": {
                    "BONDPARAMS": {
                        "O H": 0.08,
                    },
                },
                "CI": {"SIZECI": 2},
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.64172767, atol=1e-7)

    @pytest.mark.xfail(reason="NOT CRITICAL -> TO FIX")
    @pytest.mark.forces
    def test_ci_cm3_grad(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True, "DISP": 2},
                "CM3": {
                    "BONDPARAMS": {
                        "O H": 0.08,
                    },
                },
                "CI": {"SIZECI": 2},
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
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

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.64172767, atol=1e-7)

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
        grad = compute_numgrad(
            symbols=water.symbols, positions=water.positions, calculator=dem, delta=0.001
        )

        assert np.allclose(results["forces"], grad, atol=1e-5)

    def test_ci_wmull(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True, "DISP": 2},
                "WMULL": {
                    "BONDPARAMS": {
                        "O H": 0.08,
                    },
                },
                "CI": {"SIZECI": 2},
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.64834302, atol=1e-7)

    @pytest.mark.xfail(reason="NOT CRITICAL -> TO FIX")
    @pytest.mark.forces
    def test_ci_wmull_grad(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True, "DISP": 2},
                "WMULL": {
                    "BONDPARAMS": {
                        "O H": 0.08,
                    },
                },
                "CI": {"SIZECI": 2},
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
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

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.64834302, atol=1e-7)

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
        grad = compute_numgrad(
            symbols=water.symbols, positions=water.positions, calculator=dem, delta=0.01
        )

        assert np.allclose(results["forces"], grad, atol=1e-5)

    def test_ci_dipole(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                },
                "CI": {"SIZECI": 2},
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
                "DIPOLE": {
                    "OUTFILE": True,
                },
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        assert np.allclose(results["energy"]["energy"], -7.65086842, atol=1e-7)

        filename = os.path.join(dem.workdir, "deMon.dip")
        assert os.path.exists(filename)

        dipole = results["tensors"]["dipole_norm"]
        assert np.allclose(dipole, 1.82336646, 1e-5)

    def test_ci_mdyn(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                },
                "CI": {"SIZECI": 2},
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
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

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        pote = results["potential_energy"]
        kine = results["kinetic_energy"]
        tote = results["total_energy"]

        assert np.sum((tote - (pote + kine))[1:]) <= 1e-5

    def test_ci_mdyn_constraint(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                },
                "CI": {"SIZECI": 2},
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
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

        dem.calculate(symbols=water.symbols, positions=water.positions)

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

    def test_ci_ptmc(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                },
                "CI": {"SIZECI": 2},
                "CUTSYS": {
                    "RIGID": False,
                    "FRAGMENT": [20, 3],
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

    def test_ci_ptmc_rigid(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                },
                "CI": {"SIZECI": 2},
                "CUTSYS": {
                    "RIGID": True,
                    "FRAGMENT": [20, 3],
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
