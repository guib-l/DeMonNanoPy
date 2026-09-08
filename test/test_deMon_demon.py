from copy import deepcopy

import os
import numpy as np
from ase.atoms import Atoms

import deMonPy
from deMonPy.ase_calculator import DeMonNano
from deMonPy.deMonNano import deMonNano
from deMonPy.deMonNano import Module_DeMonNano

deMonPy.configure_from_file("global.json")

parameters = {
    "DEMON_EXECUTABLE": deMonPy.DEMON_EXECUTABLE,
    "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
    "DEMON_PARAMETERS": {
        "ACTIVE": {
            "DFTB": {"SCC": True, "DISP": 2},
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


WORKDIR = ".run/demon"


class TestBasicUsage:
    def test_single_point(self):

        parameter_config = deepcopy(parameters)

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -8.06209886

    def test_module_opt(self):

        mod = Module_DeMonNano(
            module="opt",
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **parameters,
        )

        mod(image=image, max=10)

        assert np.allclose(mod.results["energy"]["energy"],-8.1488236, atol=1e-7)

    def test_module_md(self):

        mod = Module_DeMonNano(
            module="md",
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **parameters,
        )

        mod(image=image, temp=10)

        assert np.allclose(len(mod.results["trajectory"]),11, atol=1e-1)


    def test_module_mc(self):

        mod = Module_DeMonNano(
            module="ptmc",
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **parameters,
        )

        # Run PTMC
        mod(
            method="mc",
            image=image,
            max=30,
            temperature=30
        )
        assert np.allclose(mod.results["ptmc"]["nb_temp"],1, atol=1e-1)


    def test_module_ase_sp(self):

        calc = DeMonNano(
            omp_threads=1,
            title="CALCULATION DEMONANO",
            directory=WORKDIR,
            **parameters,
        )

        calc.calculate(atoms=image,properties=["energy","forces"])
        
        assert np.allclose(calc.results["energy"],-219.3808842460711, atol=1e-7)

    def test_module_ase_bfgs(self):

        from ase.optimize import BFGS
        trajfile = os.path.join(WORKDIR,'H2O.traj')

        calc = DeMonNano(
            omp_threads=1,
            title="CALCULATION DEMONANO",
            directory=WORKDIR,
            **parameters,
        )
        image.calc = calc
        opt = BFGS(image, trajectory=trajfile)
        opt.run(fmax=0.05)

        assert os.path.exists(trajfile)














