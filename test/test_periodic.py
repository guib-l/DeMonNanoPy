import os
from copy import deepcopy

import numpy as np
from ase.atoms import Atoms

import deMonPy
from deMonPy.deMonNano import deMonNano

from deMonPy.molden import read_XYZ

from scipy.optimize import minimize

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


    def test_periodic_graphen(self):

        cell = np.array([
            [ 2.460000,  0.000000,  0.000000],
            [-1.230000,  2.130422,  0.000000],
            [ 0.000000,  0.000000,  100.0000],
        ])
        positions = np.array([
            [0.000000, 0.000000, 0.000000],
            [1.230000, 0.710000, 0.000000],
        ])
        symbols = ["C"]*2
        kpts = [10, 10, 1]

        parameter_config = deepcopy(parameters)

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=symbols, positions=positions, cell=cell, kpts=kpts)

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -3.47255046



    def test_periodic_graphite(self):

        cell = np.array([
            [ 2.460000,  0.000000,  0.000000],
            [-1.230000,  2.130422,  0.000000],
            [ 0.000000,  0.000000,  6.700000],
        ])
        positions = np.array([
            [0.000000, 0.000000, 0.000000],
            [1.230000, 0.710000, 0.000000],
        ])
        symbols = ["C"]*2
        kpts = [10, 10, 10]

        parameter_config = deepcopy(parameters)

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=symbols, positions=positions, cell=cell, kpts=kpts)

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -3.47292187


    def test_periodic_graph_optAtoms(self):

        cell = np.array([
            [ 2.460000,  0.000000,  0.000000],
            [-1.230000,  2.130422,  0.000000],
            [ 0.000000,  0.000000,  100.0000],
        ])
        positions = np.array([
            [0.000000, 0.000000, 0.000000],
            [1.230000, 0.710000, 0.000000],
        ])
        symbols = ["C"]*2
        kpts = [10, 10, 1]

        parameter_config = deepcopy(parameters)
        parameter_config.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {
                            "MAX": 99,
                            "TRAJECTORY": True,
                        },
                    },
                }
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=symbols, positions=positions, cell=cell, kpts=kpts)

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -3.47255048


    def test_periodic_graph_optCell(self):

        cell = np.array([
            [ 2.510000,  0.000000,  0.000000],
            [-1.250000,  2.110422,  0.000000],
            [ 0.000000,  0.000000,  100.0000],
        ])
        positions = np.array([
            [0.000000, 0.000000, 0.000000],
            [1.230000, 0.710000, 0.000000],
        ])
        symbols = ["C"]*2
        kpts = [10, 10, 1]

        parameter_config = deepcopy(parameters)
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        def to_minimize(factor):
            _cell = cell * factor.T

            dem.calculate(symbols=symbols, positions=positions, cell=_cell, kpts=kpts)

            results = dem.results
            return results["energy"]["energy"]

        res = minimize(to_minimize,np.array([1.,1.,1.]),method="COBYLA",options={"rhobeg":0.01})

        ref_cell = np.array([
            [ 2.460000,  0.000000,  0.000000],
            [-1.230000,  2.130422,  0.000000],
            [ 0.000000,  0.000000,  100.0000],
        ])
        new_cell = cell * res.x.T
        
        assert np.allclose(new_cell,ref_cell,rtol=0.01 )










