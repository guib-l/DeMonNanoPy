import copy

import ase
import matplotlib.pyplot as plt
import numpy as np

import ase 
from ase import Atoms

import deMonPy
from deMonPy.deMonNano import deMonNano
from deMonPy.molden import progressbar, read_XYZ

deMonPy.configure_from_file("global.json")

base_parameters = {
    "DEMON_EXECUTABLE": deMonPy.DEMON_EXECUTABLE,
    "BASIS": {"PTYPE": "MAT", "SKFILE": deMonPy.DEMON_BASIS},
    "DEMON_PARAMETERS": {
        "ACTIVE": {
            "DFTB": {"SCC": True, "DISP": 2, "FERMI": 10},
            "CHARGE": 1.0,
        },
    },
}

WORKDIR = ".run/examples-ci/"

positions = np.array(
    [
        [0.713108 ,0.000000, 0.000000 ],
        [-0.713108, 0.000000, 0.000000],
        [1.427512 ,1.232835, 0.000000 ],
        [-1.427512, 1.232835, 0.000000],
        [-1.427512, -1.232835, 0.00000],
        [1.427512 ,-1.232835, 0.000000],
        [2.831510 ,1.208118, 0.000000 ],
        [-2.831510, 1.208118, 0.000000],
        [-2.831510, -1.208118, 0.00000],
        [2.831510 ,-1.208118, 0.000000],
        [0.680302 ,2.459268 ,0.000000 ],
        [-0.680302, 2.459268 ,0.0000000],
        [-0.680302, -2.459268, 0.00000],
        [0.680302 ,-2.459268, 0.0000000],
        [3.520334 ,0.000000 ,0.000000 ],
        [-3.520334, 0.000000, 0.000000],
        [4.622312 ,0.000000 ,0.000000 ],
        [-4.622312, 0.000000 ,0.000000],
        [-3.392693, 2.156285 ,0.000000],
        [-3.392693, -2.156285, 0.000000],
        [3.392693 ,-2.156285 ,0.0000008],
        [3.392693 ,2.156285 ,0.000000 ],
        [1.230247 ,3.414570 ,0.000000 ],
        [-1.230247, -3.41457, 0.00000],
        [1.230247 ,-3.414570, 0.000000],
        [-1.230247, 3.414570, 0.000000],
    ])

def _save(fig, filename):
    fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=150)
        print(f"Figure : {filename}")


def exemple_ci_pyrene():

    atoms = Atoms( ["C"] * 16 + ["H"] * 10, positions=positions)
    tmp_atoms = atoms.copy()
    tmp_atoms.positions[:,2] = 3.4
    images = [atoms + tmp_atoms]

    parameter_config = copy.deepcopy(base_parameters)
    parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
        {
            "CM3": {
                "BONDPARAMS": {"C H": 0.10},
            },
            "CI": {
                "SIZECI": 2,
            },
            "CUTSYS": {"FRAGMENT": [26, 26]},
        }
    )
    base_parameters2 = copy.deepcopy(base_parameters)
    base_parameters2["DEMON_PARAMETERS"]["ACTIVE"].update(
        {
            "CM3": {
                "BONDPARAMS": {"C H": 0.10},
            },
        }
    )

    dist, trace, ci = [], [], []

    for r_ in progressbar(np.linspace(2.8, 14, 65), size=40, prefix=" \u23f3 Calculation : "):
        pyrenes = images[-1].copy()
        pyrenes.positions[26:, 2] = r_
        dist.append(r_)

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **base_parameters2)
        dem.calculate(symbols=pyrenes.symbols, positions=pyrenes.positions)
        energy = dem.results["energy"]
        trace.append(energy["energy"])

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
        dem.calculate(symbols=pyrenes.symbols, positions=pyrenes.positions)
        energy = dem.results["energy"]
        ci.append(energy["energy"])

    trace = np.array(trace)
    ci = np.array(ci)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(dist, (trace - ci[-1]) * ase.units.Hartree, color="black", label="DFTB")
    ax.plot(dist, (ci - ci[-1]) * ase.units.Hartree, color="red", label="DFTB-CI")
    ax.legend()
    ax.set_xlabel("Distances (A)")
    ax.set_ylabel("Energy (eV)")

    _save(fig, "./dissociation-profile.png")


def exemple_exci_pyrene():

    images, _ = read_XYZ("./example/data_test/pyrene-2.xyz")

    parameter_config = copy.deepcopy(base_parameters)
    parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
        {
            "CM3": {
                "BONDPARAMS": {"C H": 0.10},
            },
            "CI": {"SIZECI": 10, "EXCCI": 4},
            "CUTSYS": {"FRAGMENT": [26, 26]},
        }
    )

    dist, excci = [], []

    for r_ in progressbar(np.linspace(2.7, 6.0, 65), size=40, prefix=" \u23f3 Calculation : "):
        pyrenes = images[-1].copy()
        pyrenes.positions[26:, 2] = r_

        dist.append(r_)

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
        dem.calculate(symbols=pyrenes.symbols, positions=pyrenes.positions)

        n_state = len(dem.results["states"])
        results = dem.results["states"]

        _excci = []
        for ig in range(1, n_state + 1):
            _excci.append(results[f"state {ig}"]["energy"])

        excci.append(np.array(_excci))

    dist = np.array(dist)
    excci = np.array(excci).T

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    lowest = np.min((excci)[:, -1])

    for ig in range(len(excci)):
        ax.plot(dist, (excci[ig] - lowest) * ase.units.Hartree, color="black")

    ax.set_xlabel("Distances (A)")
    ax.set_ylabel("Energy (eV)")

    _save(fig, "./profile-pyrene.png")


def exemple_exci_pyrene_rot(distance=3.2):

    images, _ = read_XYZ("./example/data_test/pyrene-2.xyz")

    parameter_config = copy.deepcopy(base_parameters)
    parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
        {
            "CM3": {
                "BONDPARAMS": {"C H": 0.10},
            },
            "CI": {"SIZECI": 10, "EXCCI": 4},
            "CUTSYS": {"FRAGMENT": [26, 26]},
        }
    )

    angle, excci = [], []

    for r_ in progressbar(np.linspace(0, 90, 35), size=40, prefix=" \u23f3 Calculation : "):
        pyrenes = images[-1].copy()
        pyrenes.positions[26:, 2] = distance

        pyrene_1 = pyrenes[:26]
        pyrene_2 = pyrenes[26:]
        pyrene_2.euler_rotate(r_, 0.0, 0.0)
        pyrenes = pyrene_1 + pyrene_2

        angle.append(r_)

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
        dem.calculate(symbols=pyrenes.symbols, positions=pyrenes.positions)

        n_state = len(dem.results["states"])
        results = dem.results["states"]

        _excci = []
        for ig in range(1, n_state + 1):
            _excci.append(results[f"state {ig}"]["energy"])

        excci.append(np.array(_excci))

    angle = np.array(angle)
    excci = np.array(excci).T

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    lowest = np.min((excci)[:, -1])

    for ig in range(len(excci)):
        ax.plot(angle, (excci[ig] - lowest) * ase.units.Hartree, color="black")

    ax.set_xlabel("Angle (degree)")
    ax.set_ylabel("Energy (eV)")

    _save(fig, f"./profile-pyrene_rot-{distance}.png")


if __name__ == "__main__":
    exemple_ci_pyrene()

    exemple_exci_pyrene()

    exemple_exci_pyrene_rot(distance=3.09)

    exemple_exci_pyrene_rot(distance=3.50)
