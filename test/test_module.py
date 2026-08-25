import copy
import os

import numpy as np
from ase.atoms import Atoms

import deMonPy
from deMonPy.deMonNano import Module_DeMonNano

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

WORKDIR = ".run/modules"


class TestOptimization:
    def test_opt(self):

        copy_parameters = copy.deepcopy(parameters)
        mod = Module_DeMonNano(
            module="opt", title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters
        )

        mod(image=image, max=10)

        results = mod.results
        assert np.allclose(results["energy"]["energy"], -8.14876390, atol=1e-7)


class TestMonteCarlo:
    def test_mc(self):
        pass


class TestPTMC:
    def test_ptmc(self):
        pass


class TestDynamics:
    def test_dyn(self):
        pass
