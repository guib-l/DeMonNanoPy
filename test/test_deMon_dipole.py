import os
from copy import deepcopy

import numpy as np
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
        },
    },
}

image = Atoms(
    [
        "O",
        "H",
        "H",
    ],
    positions=np.array(
        [
            [1.2478, -0.5185, 3.4049],
            [1.5946, -1.4204, 3.3886],
            [0.9008, -0.3341, 2.5062],
        ]
    ),
)

WORKDIR = ".run/dipole/"


class TestFreq:
    def test_dipole_file(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DIPOLE": {
                    "OUTFILE": True,
                },
            }
        )
        dem = deMonNano(
            title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config
        )
        dem.clean_workdir()

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        assert np.allclose(results["energy"]["energy"], -4.07497884, atol=1e-7)

        filename = os.path.join(dem.workdir, "deMon.dip")
        assert os.path.exists(filename)

        dipole = results["tensors"]["dipole_norm"]
        assert np.allclose(dipole, 0.62607342, 1e-5)

    def test_dipole_water(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DIPOLE": {
                    "OUTFILE": False,
                },
                "WMULL": {
                    "BONDPARAMS": {
                        "O H": 0.18,
                    },
                },
            }
        )
        dem = deMonNano(
            title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config
        )
        dem.clean_workdir()

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        assert np.allclose(results["energy"]["energy"], -4.06678665, atol=1e-7)

        filename = os.path.join(dem.workdir, "deMon.dip")
        assert os.path.exists(filename)

        dipole = results["tensors"]["dipole_norm"]
        assert np.allclose(dipole, 0.71893185, 1e-5)
