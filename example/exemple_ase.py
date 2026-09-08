import os
import numpy as np
from ase.atoms import Atoms

import deMonPy
from deMonPy.ase_calculator import DeMonNano

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

WORKDIR = ".run/ase"


def exemple_run_ase():

    calc = DeMonNano(
        omp_threads=1,
        title="CALCULATION DEMONANO",
        directory=WORKDIR,
        **parameters,
    )

    calc.calculate(atoms=image,properties=["energy","forces"])

    print(calc.results)


def exemple_run_bfgs():

    from ase.optimize import BFGS

    calc = DeMonNano(
        omp_threads=1,
        title="CALCULATION DEMONANO",
        directory=WORKDIR,
        **parameters,
    )
    image.calc = calc
    opt = BFGS(image, trajectory=os.path.join(WORKDIR,'H2O.traj'))
    opt.run(fmax=0.005)


if __name__ == "__main__":
    exemple_run_ase()

    exemple_run_bfgs()
