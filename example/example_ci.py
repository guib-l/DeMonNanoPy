import copy

import ase
import matplotlib.pyplot as plt
import numpy as np

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


def _save(fig, filename):
    fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=150)
        print(f"Figure : {filename}")


def exemple_ci_pyrene():

    images, _ = read_XYZ("./example/data_test/pyrene-2_cation.xyz")

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

    dist, trace, ci = [], [], []

    for r_ in progressbar(np.linspace(0.75, 3.0, 65), size=40, prefix=" \u23f3 Calculation : "):
        pyrenes = images[-1].copy()
        pyrenes.positions[26:, 2] *= r_
        dist.append((pyrenes.positions[26, 2] * r_) - pyrenes.positions[0, 2])

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **base_parameters)
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
