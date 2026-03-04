
import pytest
import configs
import numpy as np

import copy
from copy import deepcopy

from ase.atoms import Atoms
from deMonPy.deMonNano import deMonNano

parameters = {
    "DEMON_EXECUTABLE":configs.EXECUTABLE,
    "BASIS":{
        "PTYPE":"BIO",
        "SKFILE":"../basis"
    },
    "DEMON_PARAMETERS":{
        "ACTIVE":{
            "DFTB":{
                "SCC":True,
            },
        },
    }
}

image = Atoms(
    ["O","H","H","O","H","H"],
    positions=np.array(
        [[1.2478,-0.5185,3.4049],
        [1.5946,-1.4204,3.3886],
        [0.9008,-0.3341,2.5062],
        [3.2478,-0.4185,3.4049],
        [3.5946,-1.5204,3.3886],
        [2.9008,-0.3341,2.6062],
        ])
    )

WORKDIR = ".run/"





class TestBasicUsage:

    def test_single_point(self):

        parameter_config = deepcopy(parameters)

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

    def test_bondparams(self):

        parameter_config = deepcopy(parameters)
        parameter_config['DEMON_PARAMETERS']['ACTIVE'].update(
            {"WMULL":{
                "BONDPARAMS":{
                    "N C":0.39,
                    "C H":0.48,
                    "N H":0.60,
                    "O H":0.18,
                    "O C":0.0,
                    "O N":0.0 },
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

        assert energy["energy"] == -8.06209343
        assert energy["electronic_energy"] == -8.21888334
        assert energy["coulomb_energy"] == 0.04185358
        assert energy["repulsive_energy"] == 0.15678992

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

    def test_cutsys(self):

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

    def test_debug(self):

        parameter_config = deepcopy(parameters)
        parameter_config['DEMON_PARAMETERS']['ACTIVE'].update(
            {
                "PRINT":{"DEBUG":True},
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


class TestDftbUsage:

    def test_freq(self):

        parameter_config = deepcopy(parameters)
        parameter_config['DEMON_PARAMETERS']['ACTIVE'].update(
            {"FREQ":False,}
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

    def test_tddftb(self):

        parameter_config = deepcopy(parameters)
        parameter_config['DEMON_PARAMETERS']['ACTIVE'].update(
            {"TD-DFTB":True,}
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

        assert 'triplet' in results.keys()
        assert 'singlet' in results.keys()

    def test_ci(self):

        parameter_config = deepcopy(parameters)
        parameter_config['DEMON_PARAMETERS']['ACTIVE'].update(
            {
                "CI":{
                    "SIZECI":2,
                },
                "CUTSYS":{
                    "FRAGMENT":[3,3],
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
        conf_1 = results["configuration_1"]
        conf_2 = results["configuration_2"]
        
        assert conf_1["energy"] == -8.06209342
        assert conf_2["energy"] == -8.06209342

    def test_const(self):

        parameter_config = deepcopy(parameters)
        parameter_config['DEMON_PARAMETERS']['ACTIVE'].update(
            {
                "CI":{
                    "CONST":1
                },
                "CUTSYS":{
                    "FRAGMENT":[3,3],
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
        
        assert energy["energy"] == -8.06209342
        assert energy["electronic_energy"] == -8.21888334
        assert energy["coulomb_energy"] == 0.04185370
        assert energy["repulsive_energy"] == 0.15678992





