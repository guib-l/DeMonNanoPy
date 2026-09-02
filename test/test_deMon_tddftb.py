import os
from copy import deepcopy

import numpy as np
from ase.atoms import Atoms

import shutil
import deMonPy
from deMonPy.deMonNano import deMonNano

deMonPy.configure_from_file("global.json")

parameters = {
    "DEMON_EXECUTABLE": deMonPy.DEMON_EXECUTABLE,
    "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
    "DEMON_PARAMETERS": {
        "ACTIVE": {
            "DFTB": {"SCC": True},
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


WORKDIR = ".run/tddftb/"


class TestTDDFTB:
    def test_tddftb_closeShell(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "TD-DFTB": True,
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -8.06209343
        assert energy["electronic_energy"] == -8.21888334
        assert energy["coulomb_energy"] == 0.04185358
        assert energy["repulsive_energy"] == 0.15678992

        assert "triplet" in results.keys()
        assert "singlet" in results.keys()

    def test_tddftb_dftb3(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True, "THIRD":True, "GCOR":4.0},
                "TD-DFTB": True,
            }
        )
        shutil.copy2("test/data_test/3ord_param", f"{WORKDIR}/3ord_param")
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -8.08604681, atol=1e-7)
        assert np.allclose(energy["electronic_energy"], -8.24283673, atol=1e-7)
        assert np.allclose(energy["coulomb_energy"], 0.03744967, atol=1e-7)
        assert np.allclose(energy["repulsive_energy"], 0.15678992, atol=1e-7)

        assert "triplet" in results.keys()
        assert "singlet" in results.keys()

    def test_tddftb_fermi(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True, "FERMI":50, },
                "TD-DFTB": True,
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -8.06209343, atol=1e-7)

        assert "triplet" in results.keys()
        assert "singlet" in results.keys()

    def test_tddftb_ldep(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True, "L-DEP":True, },
                "TD-DFTB": True,
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -8.06480023, atol=1e-7)

        assert "triplet" in results.keys()
        assert "singlet" in results.keys()

    def test_tddftb_disp(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True, "DISP":2, },
                "TD-DFTB": True,
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -8.06209886, atol=1e-7)

        assert "triplet" in results.keys()
        assert "singlet" in results.keys()








    def test_tddftb_mdlresp(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {"DFTB": {"SCC": True}, "TD-DFTB": {"FRESP": 10}}
        )
        parameter_config["DEMON_MODULE"] = {"ACTIVE": {}}
        parameter_config["DEMON_MODULE"]["ACTIVE"].update(
            {
                "MD": {
                    "MDYNAMICS": {
                        "RANDOM": 300,
                    },
                    "TIMESTEP": 0.5,
                    "MDSTEP": {"MAX": 50, "OUT": 1},
                    "TRAJECTORY": True,
                }
            }
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results

        pote = results["potential_energy"]
        kine = results["kinetic_energy"]
        tote = results["total_energy"]

        assert np.sum((tote - (pote + kine))[1:]) <= 1e-5
        traj = results["trajectory"]
        temperature = [atm.get_temperature() for atm in traj]

        assert (temperature[0] - 300.0) < 0.02

        filename = os.path.join(dem.workdir, "spectra_detailed.out")
        assert os.path.exists(filename)

        filename = os.path.join(dem.workdir, "spectra.out")
        assert os.path.exists(filename)

        filename = os.path.join(dem.workdir, "spectrum_final.out")
        assert os.path.exists(filename)
