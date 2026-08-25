import copy
import os

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
            "DFTB": {
                "SCC": True,
                "DISP": 2,
                "MAX": 1000,
                "TOL": 1e-7,
            },
            "CM3": {"BONDPARAMS": {"C H": 0.14, "O H": 0.13, "O C": 0.0}},
        },
    },
}


image = Atoms(
    [
        "O",
        "H",
        "H",
    ]
    + ["C"] * 24
    + ["H"] * 12,
    positions=np.array(
        [
            [0.61440, -3.128685, 1.261524],
            [-0.079491, -2.549303, 1.606524],
            [1.294900, -2.551439, 0.887481],
            [0.052302, -0.009515, 1.641099],
            [0.052225, 0.052477, 3.060236],
            [1.288787, -0.018542, 0.927144],
            [2.518486, 0.034562, 1.636286],
            [-1.183277, -0.068207, 0.928740],
            [-2.413067, -0.063742, 1.639357],
            [1.288882, -0.085895, -0.498679],
            [2.519058, -0.099369, -1.208400],
            [0.053279, -0.144740, -1.210209],
            [0.053888, -0.215361, -2.628596],
            [-1.182430, -0.135986, -0.496718],
            [-2.411825, -0.198190, -1.204880],
            [3.740023, -0.044293, -0.474443],
            [3.739905, 0.020302, 0.901037],
            [-3.633173, -0.191170, -0.469555],
            [-3.633965, -0.126197, 0.905835],
            [1.301313, -0.226023, -3.318715],
            [2.493501, -0.170214, -2.632078],
            [-1.193035, -0.275139, -3.317116],
            [-2.385445, -0.266953, -2.628614],
            [1.299489, 0.104668, 3.748856],
            [2.491857, 0.096107, 3.060435],
            [-1.195211, 0.055974, 3.750505],
            [-2.387194, 0.000185, 3.063368],
            [4.691379, -0.054769, -1.025283],
            [-4.584085, -0.239952, -1.019148],
            [1.300172, -0.281342, -4.416744],
            [-1.191117, -0.329632, -4.415179],
            [-1.193686, 0.103622, 4.848757],
            [1.297489, 0.151248, 4.847161],
            [3.446477, -0.180799, -3.180139],
            [-3.337922, -0.315062, -3.175565],
            [-3.339897, 0.003843, 3.612013],
            [3.444496, 0.135878, 3.607602],
            [-4.585108, -0.122647, 1.457164],
            [4.690818, 0.061300, 1.451086],
        ]
    ),
)


WORKDIR = ".run/water_coron/"


class TestWaterCoronene:
    def test_coronene(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {
                            "MAX": 99999,
                            "TOL": 3e-5,
                            "GRADTOL": 5e-5,
                            "OUT": 10,
                            "TRAJECTORY": True,
                        },
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols[3:], positions=image.positions[3:])

        results = mod.results
        last = results["output_geometry"]

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(FREQ=True)

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=last.symbols, positions=last.positions)

        results = mod.results

        modes = [mod["frequency"] for mod in results["frequency"]]

        indices = np.where(np.isclose(modes, 824, atol=1.0))[0]
        assert len(indices) > 0
        indices = np.where(np.isclose(modes, 2977, atol=1.0))[0]
        assert len(indices) > 0
        indices = np.where(np.isclose(modes, 2994, atol=1.0))[0]
        assert len(indices) > 0

    @pytest.mark.optional
    def test_water_coronene(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {
                            "MAX": 99999,
                            "TOL": 1e-7,
                            "GRADTOL": 1e-5,
                            "OUT": 1,
                            "TRAJECTORY": False,
                        },
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        last = results["output_geometry"]

        assert np.allclose(results["energy"]["energy"], -49.96219836, atol=1e-7)

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(FREQ=True)

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=last.symbols, positions=last.positions)

        results = mod.results

        modes = [mod["frequency"] for mod in results["frequency"]]

        frequencies = [3771, 1581, 4038, 827]
        for vib in frequencies:
            indices = np.where(np.isclose(modes, vib, atol=1.0))[0]
            assert len(indices) > 0, f"Frequency {vib} didn't exists"
