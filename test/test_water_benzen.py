import copy
import os

# import configs
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
            "DFTB": {
                "SCC": True,
                "DISP": 2,
                "MAX": 1000,
                "TOL": 1e-5,
            },
            "CM3": {"BONDPARAMS": {"C H": 0.098, "O H": 0.13, "O C": 0.0}},
        },
    },
}

image = Atoms(
    [
        "O",
        "H",
        "H",
    ]
    + [
        "C",
        "H",
        "C",
        "C",
        "C",
        "C",
        "C",
        "H",
        "H",
        "H",
        "H",
        "H",
    ],
    positions=np.array(
        [
            [0.071715, 0.071715, 2.500000],
            [1.016642, -0.088358, 2.600000],
            [-0.088358, 1.016642, 2.700000],
            [0.000000, 1.401045, 0.000000],
            [0.000000, 2.409045, 0.000000],
            [1.212436, 0.701045, 0.000000],
            [1.212436, -0.698955, 0.000000],
            [0.000000, -1.398955, 0.000000],
            [-1.212436, -0.698955, 0.000000],
            [-1.212436, 0.701045, 0.000000],
            [2.155537, 1.245545, 0.000000],
            [2.155537, -1.243455, 0.000000],
            [0.000000, -2.487955, 0.000000],
            [-2.155537, -1.243455, 0.000000],
            [-2.155537, 1.245545, 0.000000],
        ]
    ),
)


WORKDIR = ".run/water_benz/"


class TestWaterBenzen:
    def test_benzen(self):

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

        mod = deMonNano(
            title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters
        )

        mod.calculate(symbols=image.symbols[3:], positions=image.positions[3:])

        results = mod.results

        last = results["output_geometry"]

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(FREQ=True)

        mod = deMonNano(
            title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters
        )

        mod.calculate(symbols=last.symbols, positions=last.positions)

        results = mod.results

        modes = [mod["frequency"] for mod in results["frequency"]]

        assert np.allclose(modes[4], 657, atol=1.0)
        assert np.allclose(modes[27], 3012, atol=1.0)
        assert np.allclose(modes[28], 3013, atol=1.0)

    @pytest.mark.optional
    def test_water_benzen(self):

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

        mod = deMonNano(
            title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters
        )

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results

        last = results["output_geometry"]

        o1_positions_1 = last.positions[0]
        h1_positions_1 = last.positions[1]
        h1_positions_2 = last.positions[2]
        c_positions_1 = last.positions[6]
        c_positions_2 = last.positions[9]

        assert np.allclose(last.get_angles([[1, 0, 2]]), 108.1, atol=2e-1)

        assert np.allclose(
            np.linalg.norm(h1_positions_1 - c_positions_1), 2.38, atol=2e-2
        )
        assert np.allclose(
            np.linalg.norm(h1_positions_2 - c_positions_2), 3.39, atol=2e-2
        )
        assert np.allclose(
            np.linalg.norm(o1_positions_1 - h1_positions_1), 0.96, atol=1e-2
        )

        assert np.allclose(results["energy"]["energy"], -16.61297421, atol=1e-7)
        assert np.allclose(results["energy"]["london_energy"], -0.00158364, atol=1e-7)

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(FREQ=True)

        mod = deMonNano(
            title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters
        )

        mod.calculate(symbols=last.symbols, positions=last.positions)

        results = mod.results

        modes = [mod["frequency"] for mod in results["frequency"]]

        assert np.allclose(modes[10], 657 + 4, atol=1.0)
        assert np.allclose(modes[31], 3012 - 34, atol=1.0)
        assert np.allclose(modes[32], 3012 - 33, atol=1.0)
        assert np.allclose(modes[36], 3013 + 18, atol=1.0)
