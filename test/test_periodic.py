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
            "DFTB": {"SCC": True, "DISP": 2},
            "PBC": {
                "TYPE": "GENERAL",
                "LATTICE": np.array([[10, 0.0, 0.0], [0.0, 11, 0], [0.0, 0.0, 12]]),
                "KPTS": [4, 4, 3],
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


WORKDIR = ".run/pbc"


class TestPeriodic:
    def test_periodic_basic(self):

        parameter_config = deepcopy(parameters)

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -8.062359

        dem.print_results()

        # from ase.visualize import view
        # view(dem.results['output_geometry'])
