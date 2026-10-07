import copy

import matplotlib.pyplot as plt
import numpy as np

import deMonPy
from deMonPy.deMonNano import deMonNano
from deMonPy.molden import read_XYZ

deMonPy.configure_from_file("global.json")


parameters = {
    "DEMON_EXECUTABLE": deMonPy.DEMON_EXECUTABLE,
    "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
    "DEMON_PARAMETERS": {
        "ACTIVE": {
            "DFTB": {"SCC": True},
            "CHARGE": 0.0,
        },
    },
}

WORKDIR = ".run/examples-argon/"


def _save(fig, filename):
    fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=150)
        print(f"Figure : {filename}")


def example_argon_cluster():
    parameter_config = copy.deepcopy(parameters)
    parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
        {
            "DFTB": {
                "SCC": True,
                "DISP": 2,
                "POLA": True,
                "NOPOLQM": True,
            },
            "RG": {"COUPLING": "ARGON", "ALPHARG": 11.07, "FILENAME": None},
        }
    )
    images, ref = read_XYZ("example/data_test/argon-test.mol")

    size, energies = [], []
    for i, image in enumerate(images):
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
        dem.calculate(symbols=image.symbols, positions=image.positions)
        results = dem.results

        size.append(len(image) - 20)
        energies.append(results["energy"]["energy"])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    energies = np.array(energies)
    size = np.array(size)

    ax.plot(size[1:], ((energies[0] - energies[1:]) * 219474) / size[1:], color="black", marker="s")

    ax.set_xlabel("Number of argon atoms")
    ax.set_ylabel(r"Energy (cm$^{-1}$)")

    _save(fig, "./DABCO-argons.png")


if __name__ == "__main__":
    example_argon_cluster()
