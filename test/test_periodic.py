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
                "LATTICE": np.array([[2.46, 0.0, 0.0], [-1.230000,  2.130422, 0], [0.0, 0.0, 100.]]),
                "KPTS": [4, 4, 3],
            },
        },
    },
}



WORKDIR = ".run/pbc"


class TestPeriodic:


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

        assert energy["energy"] == -3.47255044


    def test_periodic_graphen_bis(self):

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

        dem.calculate(symbols=symbols, positions=positions)

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -3.47255044


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

        assert energy["energy"] == -3.47292185


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

        assert energy["energy"] == -3.47255046


    def _test_periodic_graph_optCell(self):

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



    def test_periodic_graphen_layer(self):

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

        from ase.visualize import view
        view(Atoms(symbols,positions,cell=cell,pbc=True).repeat((10,10,1)))

        parameter_config = deepcopy(parameters)

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        energies = []

        for kpt in range(1,25):
            
            kpts = [kpt,kpt,1]
            dem.calculate(symbols=symbols, positions=positions, cell=cell, kpts=kpts)

            results = dem.results
            energy = results["energy"]

            energies.append(energy["energy"])

        print(energies)








