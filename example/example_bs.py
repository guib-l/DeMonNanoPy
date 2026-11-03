import copy

import matplotlib.pyplot as plt
import numpy as np
from ase.atoms import Atoms

import deMonPy
from deMonPy.deMonNano import deMonNano

import ase


deMonPy.configure_from_file("global.json")

parameters = {
    "DEMON_EXECUTABLE": deMonPy.DEMON_EXECUTABLE,
    "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
    "DEMON_PARAMETERS": {
        "ACTIVE": {
            "DFTB": {"SCC": True, "DISP": 2, "FERMI": 50},
            "PBC": {
                "TYPE": "GENERAL",
                "LATTICE": np.array(
                    [[2.46, 0.0, 0.0], [-1.230000, 2.130422, 0], [0.0, 0.0, 100.0]]
                ),
                "KPTS": [20, 20, 1],
            },
        },
    },
}




def exemple_run_bs():

    WORKDIR = ".run/example-bs"

    print("=" * 40)
    print(" K-points vs energy definiton")

    cell = np.array(
        [
            [2.460000, 0.000000, 0.000000],
            [-1.230000, 2.130422, 0.000000],
            [0.000000, 0.000000, 100.0000],
        ]
    )
    positions = np.array(
        [
            [0.000000, 0.000000, 0.000000],
            [1.230000, 0.710000, 0.000000],
        ]
    )
    symbols = ["C"] * 2
    kpts = [20, 20, 1]
    parameter_config = copy.deepcopy(parameters)
    dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
    energies = []

    dem.calculate(symbols=symbols, positions=positions, cell=cell, kpts=kpts)
    results = dem.results

    print(results["energy"])
    print(results["eigenvalues"])
    

if __name__ == "__main__":

    exemple_run_bs()







