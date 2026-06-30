import os

import numpy as np

import copy
from copy import deepcopy

from ase.atoms import Atoms

import deMonPy
from deMonPy.deMonNano import deMonNano


deMonPy.configure_from_file(os.path.join("..", "global.json"))

parameters = {
    "DEMON_EXECUTABLE":deMonPy.DEMON_EXECUTABLE,
    "BASIS":{
        "PTYPE":"BIO",
        "SKFILE":deMonPy.DEMON_BASIS
    },
    "DEMON_PARAMETERS":{
        "ACTIVE":{
            "DFTB":{
                "SCC":True,
                "DISP":2
            },
        },
    }
}

image = Atoms(
    ["O","H","H","O","H","H"],
    positions=np.array(
        [[1.2478,-0.5185,3.4049],
        [1.5946,-1.4204,3.3886],
        [0.9008,-0.3341,2.5062],
        [3.2478,-0.4185,3.4049],
        [3.5946,-1.5204,3.3886],
        [2.9008,-0.3341,2.6062],
        ])
    )




WORKDIR = ".run/demon"


class TestBasicUsage:

    def _test_single_point(self):

        parameter_config = deepcopy(parameters)

        dem = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **parameter_config
        )

        dem.calculate(
            symbols=image.symbols,
            positions=image.positions
        )

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -8.06209343
        assert energy["electronic_energy"] == -8.21888334
        assert energy["coulomb_energy"] == 0.04185358
        assert energy["repulsive_energy"] == 0.15678992





