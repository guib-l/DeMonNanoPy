import copy

import matplotlib.pyplot as plt
import numpy as np
from ase.atoms import Atoms

import deMonPy
from deMonPy.deMonNano import deMonNano

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
                "KPTS": [10, 10, 1],
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

WORKDIR = ".run/examples-pbc/"


def _save(fig, filename):
    fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=150)
        print(f"Figure : {filename}")


def exemple_run_kpt():

    print("="*40)
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
    kpts = [10, 10, 1]
    parameter_config = copy.deepcopy(parameters)
    dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
    energies = []

    for kpt in range(1, 25):
        kpts = [kpt, kpt, 1]
        dem.calculate(symbols=symbols, positions=positions, cell=cell, kpts=kpts)
        results = dem.results
        energy = results["energy"]
        energies.append(energy["energy"])

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    energy = np.array(energies)

    ax.plot(range(1, 25), np.abs(energy - energy[-1]), color="black", marker="s")

    ax.set_xlabel("K-points")
    ax.set_ylabel(r"|$\Delta$ Energy| (eV)")
    _save(fig, "./energy-vs-kpts.png")


def scan_graphite_D2():

    print("="*40)
    print(" SCAN of energy along z-axis : graphite-D2")

    cell = np.array(
        [
            [2.468179, 0.000000, 0.000000],
            [-1.234091, 2.1375, 0.000000],
            [0.000000, 0.000000, 6.260000],
        ]
    )
    positions = np.array(
        [
            [0.000000, 0.000000, 0.000000],
            [1.234091, 0.7125, 0.000000],
            [0.000000, 1.425000, 3.130000],
            [1.234091, 2.1375, 3.130000],
        ]
    )
    symbols = ["C"] * 4

    parameter_config = copy.deepcopy(parameters)
    parameter_config["DEMON_PARAMETERS"]["ACTIVE"]["DFTB"].update({"DISP": 2, "DIAG": "DSYGVD"})
    kpts = [17, 17, 3]
    dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

    distance, energy = [], []

    base = 6.24

    for i in range(0, 7):
        new_cell = cell.copy()
        new_positions = positions.copy()
        new_cell[2, 2] = base + (0.01 * i)
        new_positions[2:, 2] = (base + (0.01 * i)) / 2

        dem.calculate(symbols=symbols, positions=new_positions, cell=new_cell, kpts=kpts)

        results = dem.results
        print(f" > Energy total at {(base + (0.01 * i)) / 2} A : {results['energy']['energy']} Ha")

        distance.append((base + (0.01 * i)) / 2)
        energy.append(results["energy"]["energy"])

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    ax.plot(distance, energy, color="black", marker="s")

    ax.set_xlabel(r"Distance inter-layer ($\AA$)")
    ax.set_ylabel("Energy (Ha)")

    _save(fig, "./graphene-D2.png")


def scan_graphite_D1():

    print("="*40)
    print(" SCAN of energy along z-axis : graphite-D1")

    cell = np.array(
        [
            [2.47558654, 0.000000, 0.000000],
            [-1.23779327, 2.1439125, 0.000000],
            [0.000000, 0.000000, 6.760000],
        ]
    )
    positions = np.array(
        [
            [0.000000, 0.000000, 0.000000],
            [1.23779327, 0.71463749, 0.000000],
            [0.000000, 1.425000, 3.380000],
            [1.23779327, 2.14363749, 3.380000],
        ]
    )
    symbols = ["C"] * 4

    parameter_config = copy.deepcopy(parameters)
    parameter_config["DEMON_PARAMETERS"]["ACTIVE"]["DFTB"].update({"DISP": 1, "DIAG": "DSYGVD"})
    kpts = [17, 17, 3]
    dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

    distance, energy = [], []

    base = 6.74
    for i in range(0, 7):
        new_cell = cell.copy()
        new_positions = positions.copy()
        new_cell[2, 2] = base + (0.01 * i)
        new_positions[2:, 2] = (base + (0.01 * i)) / 2

        dem.calculate(symbols=symbols, positions=new_positions, cell=new_cell, kpts=kpts)

        results = dem.results
        print(f" > Energy total at {(base + (0.01 * i)) / 2} A : {results['energy']['energy']} Ha")

        distance.append((base + (0.01 * i)) / 2)
        energy.append(results["energy"]["energy"])

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    ax.plot(distance, energy, color="black", marker="s")

    ax.set_xlabel(r"Distance inter-layer ($\AA$)")
    ax.set_ylabel("Energy (Ha)")

    _save(fig, "./graphene-D1.png")


def example_periodic_graphene():

    print("="*40)
    print(" Example single-point graphene")

    cell = np.array(
        [
            [2.460000, 0.000000, 0.000000],
            [-1.230000, 2.130422, 0.000000],
            [0.000000, 0.000000, 6.700000],
        ]
    )
    positions = np.array(
        [
            [0.000000, 0.000000, 0.000000],
            [1.230000, 0.710000, 0.000000],
        ]
    )
    symbols = ["C"] * 2
    kpts = [10, 10, 10]
    parameter_config = copy.deepcopy(parameters)

    dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
    dem.calculate(symbols=symbols, positions=positions, cell=cell, kpts=kpts)

    results = dem.results
    print(" Simple point calculation")
    print(f" > Energy total : {results['energy']['energy']}")


def example_opt_graphene_D2():
    print("="*40)
    print(" Optimization Cell/Positions for graphene-D2")

    from scipy.optimize import minimize

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
    kpts = [10, 10, 1]
    parameter_config = copy.deepcopy(parameters)
    dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

    maxiter = 100

    def to_minimize(factor):
        _cell = cell * factor.T
        _positions = positions * np.array([factor])
        dem.calculate(symbols=symbols, positions=_positions, cell=_cell, kpts=kpts)
        results = dem.results
        return results["energy"]["energy"]

    res = minimize(
        to_minimize,
        np.array([1.0, 1.0, 1.0]),
        method="COBYLA",
        options={"rhobeg": 0.01, "maxiter": maxiter},
    )
    new_cell = cell * res.x.T
    new_positions = positions * np.array([res.x])

    dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
    dem.calculate(symbols=symbols, positions=new_positions, cell=new_cell, kpts=kpts)

    results = dem.results
    print(f" Optimal cell (stop after {maxiter} iterations)")
    print(f" > Energy total : {results['energy']['energy']}")

    atoms = Atoms(symbols, positions=new_positions, cell=new_cell).repeat((10, 10, 1))

    from scipy.spatial.distance import cdist

    dist = cdist(atoms.positions, atoms.positions).ravel()
    idx = np.where(dist > 0)[0]
    distance_min_CC = np.min(dist[idx])

    print(f" > Distance minimal C-C (graphene with DISP=2) : {distance_min_CC}")


def example_opt_graphene_D1():
    print("="*40)
    print(" Optimization Cell/Positions for graphene-D1")

    from scipy.optimize import minimize

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
    kpts = [10, 10, 1]
    parameter_config = copy.deepcopy(parameters)
    parameter_config["DEMON_PARAMETERS"]["ACTIVE"]["DFTB"].update({"DISP": 1})
    dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

    maxiter = 100

    def to_minimize(factor):
        _cell = cell * factor.T
        _positions = positions * np.array([factor])
        dem.calculate(symbols=symbols, positions=_positions, cell=_cell, kpts=kpts)
        results = dem.results
        return results["energy"]["energy"]

    res = minimize(
        to_minimize,
        np.array([1.0, 1.0, 1.0]),
        method="COBYLA",
        options={"rhobeg": 0.01, "maxiter": maxiter},
    )
    new_cell = cell * res.x.T
    new_positions = positions * np.array([res.x])

    dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
    dem.calculate(symbols=symbols, positions=new_positions, cell=new_cell, kpts=kpts)

    results = dem.results
    print(f" Optimal cell (stop after {maxiter} iterations)")
    print(f" > Energy total : {results['energy']['energy']}")

    atoms = Atoms(symbols, positions=new_positions, cell=new_cell).repeat((10, 10, 1))

    from scipy.spatial.distance import cdist

    dist = cdist(atoms.positions, atoms.positions).ravel()
    idx = np.where(dist > 0)[0]
    distance_min_CC = np.min(dist[idx])

    print(f" > Distance minimal C-C (graphene with DISP=1) : {distance_min_CC}")


if __name__ == "__main__":
    
    exemple_run_kpt()

    example_periodic_graphene()

    example_opt_graphene_D1()

    example_opt_graphene_D2()

    scan_graphite_D1()

    scan_graphite_D2()
