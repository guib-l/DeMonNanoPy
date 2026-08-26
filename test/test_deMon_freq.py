import os
import shutil
from copy import deepcopy

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

WORKDIR = ".run/freq/"


class TestFreq:
    def test_freq(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "FREQ": True,
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
        dem.clean_workdir()
        os.path.join(dem.workdir, "deMon.freq")

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        assert np.allclose(results["zpe"], 0.0475, atol=1e-4)
        assert len(results["frequency"]) == 18
        mode_1 = results["frequency"][0]
        assert mode_1["mode"] == 1
        assert np.allclose(mode_1["frequency"], -1178.9, atol=1e-1)
        assert np.allclose(mode_1["intensity"], 112.0, atol=1e-1)

    @pytest.mark.xfail(reason="NOT CRITICAL -> TO FIX")
    def test_limited_freq(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "FREQ": {"VIB": 5},
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
        dem.clean_workdir()

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        assert np.allclose(results["zpe"], 0.0475, atol=1e-4)
        assert len(results["frequency"]) == 11
        mode_1 = results["frequency"][0]
        assert mode_1["mode"] == 1
        assert np.allclose(mode_1["frequency"], -1178.9, atol=1e-1)
        assert np.allclose(mode_1["intensity"], 112.0, atol=1e-1)

    @pytest.mark.beta
    def test_freq_const(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "FREQ": True,
                "CI": {"CONST": 1},
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
        dem.clean_workdir()
        os.path.join(dem.workdir, "deMon.freq")

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        assert np.allclose(results["zpe"], 0.0477, atol=1e-4)
        assert len(results["frequency"]) == 18
        mode_1 = results["frequency"][0]
        assert mode_1["mode"] == 1
        assert np.allclose(mode_1["frequency"], -1134.2, atol=1e-1)
        assert np.allclose(mode_1["intensity"], 79.6, atol=1e-1)

    def test_freq_disp(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                    "DISP": 2,
                },
                "FREQ": True,
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
        dem.clean_workdir()
        os.path.join(dem.workdir, "deMon.freq")

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        assert np.allclose(results["zpe"], 0.0475, atol=1e-4)
        assert len(results["frequency"]) == 18
        mode_1 = results["frequency"][0]
        assert mode_1["mode"] == 1
        assert np.allclose(mode_1["frequency"], -1178.9, atol=1e-1)
        assert np.allclose(mode_1["intensity"], 112.0, atol=1e-1)

    @pytest.mark.xfail(reason="NOT CRITICAL -> TO FIX")
    def test_freq_ldep(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                    "L-DEP": True,
                },
                "FREQ": True,
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
        dem.clean_workdir()
        os.path.join(dem.workdir, "deMon.freq")

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        assert np.allclose(results["zpe"], 0.0475, atol=1e-4)
        assert len(results["frequency"]) == 18
        mode_1 = results["frequency"][0]
        assert mode_1["mode"] == 1
        assert np.allclose(mode_1["frequency"], -1178.9, atol=1e-1)
        assert np.allclose(mode_1["intensity"], 112.0, atol=1e-1)

    @pytest.mark.beta
    def test_freq_dftb3(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                    "THIRD": True,
                },
                "FREQ": True,
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
        dem.clean_workdir()
        os.path.join(dem.workdir, "deMon.freq")

        shutil.copy2("test/data_test/3ord_param", f"{WORKDIR}/3ord_param")
        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        assert np.allclose(results["zpe"], 0.0475, atol=1e-4)
        assert len(results["frequency"]) == 18
        mode_1 = results["frequency"][0]
        assert mode_1["mode"] == 1
        assert np.allclose(mode_1["frequency"], -1175.1, atol=1e-1)
        assert np.allclose(mode_1["intensity"], 118.7, atol=1e-1)

    def test_freq_fermi(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                    "FERMI": 150,
                },
                "FREQ": True,
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
        dem.clean_workdir()
        os.path.join(dem.workdir, "deMon.freq")

        shutil.copy2("test/data_test/3ord_param", f"{WORKDIR}/3ord_param")
        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        assert np.allclose(results["zpe"], 0.0475, atol=1e-4)
        assert len(results["frequency"]) == 18
        mode_1 = results["frequency"][0]
        assert mode_1["mode"] == 1
        assert np.allclose(mode_1["frequency"], -1178.9, atol=1e-1)
        assert np.allclose(mode_1["intensity"], 112.0, atol=1e-1)

    def test_freq_charge(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                },
                "CHARGE": 1.0,
                "FREQ": True,
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
        dem.clean_workdir()
        os.path.join(dem.workdir, "deMon.freq")

        shutil.copy2("test/data_test/3ord_param", f"{WORKDIR}/3ord_param")
        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        assert np.allclose(results["zpe"], 0.04725644, atol=1e-4)
        assert len(results["frequency"]) == 18
        mode_1 = results["frequency"][0]
        assert mode_1["mode"] == 1
        assert np.allclose(mode_1["frequency"], -1254.2, atol=1e-1)
        assert np.allclose(mode_1["intensity"], 70.7, atol=1e-1)
