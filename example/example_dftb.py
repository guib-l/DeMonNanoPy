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
            "DFTB": {"SCC": True},
            "CHARGE": 0.0,
        },
    },
}
pyrene = Atoms(
    ["C"] * 16 + ["H"] * 10,
    positions=np.array(
        [
            [0.175933, 0.335146, 1.793713],
            [1.360050, 0.325800, 1.110006],
            [-1.072593, 0.111218, 1.119476],
            [-2.302066, 0.111541, 1.806122],
            [1.397620, 0.091799, -0.306827],
            [2.606846, 0.072980, -1.028286],
            [0.172008, -0.133217, -1.000407],
            [0.181765, -0.372636, -2.406258],
            [-1.063252, -0.123546, -0.287152],
            [-2.288443, -0.353337, -0.979928],
            [-3.496740, -0.344828, -0.256738],
            [-3.495810, -0.114190, 1.119551],
            [1.412155, -0.383289, -3.091169],
            [2.606303, -0.162069, -2.403831],
            [-1.066353, -0.601267, -3.079682],
            [-2.250464, -0.592049, -2.395959],
            [-4.444118, -0.521654, -0.784767],
            [1.426319, -0.567696, -4.174401],
            [-1.053082, -0.786725, -4.163133],
            [3.557977, -0.173091, -2.952575],
            [-3.197581, -0.770057, -2.924850],
            [-4.446870, -0.110240, 1.669451],
            [0.163224, 0.514366, 2.878219],
            [2.307746, 0.497406, 1.639983],
            [3.554629, 0.245151, -0.499450],
            [-2.315834, 0.291268, 2.890140],
        ]
    ),
)
WORKDIR = ".run/examples-dftb/"


def exemple_run_dftb():

    dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameters)

    dem.calculate(symbols=pyrene.symbols, positions=pyrene.positions)

    results = dem.results

    print(" =======================================")
    print(" Draw all results : ")
    dem.print_results()

    energy = results["energy"]
    print(" =======================================")
    print(f" > Energy totale     : {energy['energy']}")
    print(f" > Energy electronic : {energy['electronic_energy']}")
    print(f" > Energy repulsion  : {energy['repulsive_energy']}")


if __name__ == "__main__":
    exemple_run_dftb()
