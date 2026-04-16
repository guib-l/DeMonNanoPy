
import os
import sys

import numpy as np

from copy import deepcopy

import pytest
from ase.atoms import Atoms

import deMonPy
from deMonPy.deMonNano import deMonNano


deMonPy.configure_from_file(os.path.join("..", "global.json"))

parameters = {
    "DEMON_EXECUTABLE":deMonPy.DEMON_EXECUTABLE,
    "BASIS":{
        "PTYPE":"BIO",
        "SKFILE":deMonPy.DEMON_BASIS
    },
    "DEMON_PARAMETERS":{
        "ACTIVE":{
            "DFTB":{
                "SCC":True
            },
        },
    }
}

image = Atoms(
    ["O","H","H","O","H","H"],
    positions=np.array([
            [1.2478,-0.5185,3.4049],
            [1.5946,-1.4204,3.3886],
            [0.9008,-0.3341,2.5062],
            [3.2478,-0.4185,3.4049],
            [3.5946,-1.5204,3.3886],
            [2.9008,-0.3341,2.6062],
        ])
    )





WORKDIR = ".run/dftbion/"


class TestCharges:

    def test_charged(self):

        parameter_config = deepcopy(parameters)
        parameter_config['DEMON_PARAMETERS']['ACTIVE'].update(
            {"CHARGE":1.0,}
        )
        dem = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **parameter_config
        )

        dem.calculate(
            symbols=image.symbols,
            positions=image.positions
        )

        results = dem.results
        energy = results["energy"]
        

        assert energy["energy"] == -7.6911342
        assert energy["electronic_energy"] == -7.84792412
        assert energy["coulomb_energy"] == 0.15950647
        assert energy["repulsive_energy"] == 0.15678992




    def test_wmull(self):

        parameter_config = deepcopy(parameters)
        parameter_config['DEMON_PARAMETERS']['ACTIVE'].update(
            {"WMULL":{
                "BONDPARAMS":{
                    "O H":0.18,},
                }
            }
        )

        dem = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **parameter_config
        )

        dem.calculate(
            symbols=image.symbols,
            positions=image.positions
        )

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -8.04442048
        assert energy["electronic_energy"] == -8.2012104
        assert energy["coulomb_energy"] == 0.05453568
        assert energy["repulsive_energy"] == 0.15678992


    def test_cm3pot(self):

        parameter_config = deepcopy(parameters)
        parameter_config['DEMON_PARAMETERS']['ACTIVE'].update(
            {"CM3":{
                "BONDPARAMS":{
                    "O H":0.08,},
                }
            }
        )

        dem = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **parameter_config
        )

        dem.calculate(
            symbols=image.symbols,
            positions=image.positions
        )

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -8.0368677
        assert energy["electronic_energy"] == -8.19365762
        assert energy["coulomb_energy"] == 0.06029717


    @pytest.mark.xfail(reason="NOT CRITICAL -> TO FIX")
    def test_cm3inter(self):

        parameter_config = deepcopy(parameters)
        parameter_config['DEMON_PARAMETERS']['ACTIVE'].update(
            {"CM3INTER":{
                "BONDPARAMS":{
                    "O H":0.08,},
                }
            }
        )

        dem = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **parameter_config
        )

        dem.calculate(
            symbols=image.symbols,
            positions=image.positions
        )

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -8.0368677



WORKDIR = ".run/dftbcutsys/"

class TestCutSys:


    def test_cutsys(self):

        parameter_config = deepcopy(parameters)
        parameter_config['DEMON_PARAMETERS']['ACTIVE'].update(
            {"CUTSYS":{
                "FRAGMENT":[3,3],
                "RIGID":False
            },
            }
        )

        dem = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **parameter_config
        )

        dem.calculate(
            symbols=image.symbols,
            positions=image.positions
        )

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -8.06209343
        assert energy["electronic_energy"] == -8.21888334
        assert energy["coulomb_energy"] == 0.04185358
        assert energy["repulsive_energy"] == 0.15678992

    def test_multiplicity(self):

        parameter_config = deepcopy(parameters)
        parameter_config['DEMON_PARAMETERS']['ACTIVE'].update(
            {
                "MULTI":1,
            }
        )

        dem = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **parameter_config
        )

        dem.calculate(
            symbols=image.symbols,
            positions=image.positions
        )

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -8.06209343
        assert energy["electronic_energy"] == -8.21888334
        assert energy["coulomb_energy"] == 0.04185358
        assert energy["repulsive_energy"] == 0.15678992






        