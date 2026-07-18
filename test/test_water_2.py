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
                "MAX": 9999,
                "TOL": 1e-5,
            },
            "CM3": {
                "BONDPARAMS": {
                    "O H": 0.13,
                }
            },
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
        "O",
        "H",
        "H",
    ],
    positions=np.array(
        [
            [0.071715, 0.071715, 3.000000],
            [1.016642, -0.088358, 2.900000],
            [-0.088358, 1.016642, 3.000000],
            [0.071715, 0.071715, 0.000000],
            [1.016642, -0.088358, 0.000000],
            [-0.088358, 1.016642, 0.000000],
        ]
    ),
)


WORKDIR = ".run/water_2/"


class TestWater2:
    def compute_relaxation(self, image):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {
                            "MAX": 99999,
                            "TOL": 1e-5,
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
        return results

    def test_water_1(self):

        results = self.compute_relaxation(image[:3])
        last = results["output_geometry"]

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(FREQ=True)

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=last.symbols, positions=last.positions)

        results = mod.results

        modes = [mod["frequency"] for mod in results["frequency"]]
        assert np.allclose(modes[1], 3804, atol=1.0)
        assert np.allclose(modes[0], 1562, atol=1.0)
        assert np.allclose(modes[2], 4078, atol=1.0)

    @pytest.mark.optional
    def test_water_2(self):

        results = self.compute_relaxation(image)
        last = results["output_geometry"]

        o1_positions_1 = last.positions[0]
        h1_positions_1 = last.positions[1]
        last.positions[2]
        o2_positions_1 = last.positions[3]
        last.positions[4]
        last.positions[5]

        assert np.allclose(last.get_angles([[1, 0, 2]]), 108.3, atol=2e-1)
        assert np.allclose(last.get_angles([[4, 3, 5]]), 109.0, atol=2e-1)

        assert np.allclose(np.linalg.norm(o2_positions_1 - h1_positions_1), 1.87, atol=1e-2)
        assert np.allclose(np.linalg.norm(o1_positions_1 - h1_positions_1), 0.96, atol=1e-2)

        assert np.allclose(results["energy"]["energy"], -8.12072071, atol=1e-7)

        copy_parameters = copy.deepcopy(parameters)

        copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(FREQ=True)

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=last.symbols, positions=last.positions)

        results = mod.results

        modes = [mod["frequency"] for mod in results["frequency"]]

        assert np.allclose(modes[1], 136, atol=1.0)
        assert np.allclose(modes[2], 153, atol=1.0)
        assert np.allclose(modes[3], 230, atol=1.0)
        assert np.allclose(modes[4], 367, atol=1.0)
        assert np.allclose(modes[5], 553, atol=1.0)
        assert np.allclose(modes[6], 1560, atol=1.0)
        assert np.allclose(modes[7], 1582, atol=1.0)
        assert np.allclose(modes[8], 3667, atol=1.0)
        assert np.allclose(modes[9], 3800, atol=1.0)
        assert np.allclose(modes[10], 3993, atol=1.0)
        assert np.allclose(modes[11], 4070, atol=1.0)
