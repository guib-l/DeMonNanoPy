import os
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
        parameter_config.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"SP": True, "TRAJECTORY": True},
                    },
                }
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        conf_1 = results["configuration_1"]
        conf_2 = results["configuration_2"]

        assert np.allclose(conf_1["energy"], -23.34473263, atol=1e-7)
        assert np.allclose(conf_2["energy"], -23.10000944, atol=1e-7)
        assert np.allclose(results["energy"]["energy"], -23.3449257719042, atol=1e-7)

        test_force = np.array(
            [
                [-0.008596062053, 0.009517421684, -0.002829737253],
                [-0.006135784494, 0.005076993325, -0.010437115548],
                [-0.000926298602, 0.001546048915, 0.002621871939],
                [0.013157986658, 0.000063826020, -0.001269807625],
                [0.010697159957, 0.004493046859, 0.006328773688],
                [0.000788079759, -0.001535814349, -0.002673232968],
                [-0.006155783352, -0.007855916017, 0.008563118171],
                [-0.003477912297, -0.012677877496, 0.000289924355],
                [0.004230149523, -0.000554331081, -0.001140410752],
                [-0.000776527291, -0.003114928315, -0.000349738304],
                [-0.001630899856, -0.001567220966, 0.002296499769],
                [0.002962422072, 0.001730979816, 0.002779901038],
                [-0.001476884097, -0.001736232973, 0.002425156231],
                [-0.002937094930, 0.003300651393, -0.000051581872],
                [-0.000630535862, -0.003263764236, -0.000192221719],
                [-0.001672341368, 0.001022905501, -0.003960389386],
                [0.003933437692, -0.000199022459, -0.001339251313],
                [-0.001227695049, 0.001165723495, -0.003798537938],
                [-0.002447740307, 0.003362402133, -0.000029238778],
                [0.002710816600, 0.002002103774, 0.002436691287],
                [-0.000879491571, -0.002155580473, 0.000976880981],
                [0.000416187302, 0.000384538277, -0.000845936415],
                [0.000074811567, 0.000994047172, 0.000198382411],
            ]
        )

        assert np.allclose(results["forces"], test_force, atol=1e-7)

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
