
import os
import sys

import numpy as np

from copy import deepcopy

from ase.atoms import Atoms
from deMonPy.deMonNano import deMonNano


EXECUTABLE = "~/Documents/dev_deMon/deMon.x"
BASIS = "~/Documents/DeMonNanoPy/test/basis"

parameters = {
    "DEMON_EXECUTABLE":EXECUTABLE,
    "BASIS":{
        "PTYPE":"BIO",
        "SKFILE":BASIS
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

WORKDIR = ".run/basics/"


class TestFreq:

    def test_freq(self):

        parameter_config = deepcopy(parameters)
        parameter_config['DEMON_PARAMETERS']['ACTIVE'].update(
            {"FREQ":True,}
        )
        dem = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=".run/freq/",
            **parameter_config
        )
        dem.clean_workdir()
        freq_file = os.path.join(dem.workdir,'deMon.freq')


        dem.calculate(
            symbols=image.symbols,
            positions=image.positions
        )

        results = dem.results
        assert results["zpe"] == 0.0475
        assert len(results["frequency"]) == 18
        mode_1 = results["frequency"][0]
        assert mode_1['mode'] == 1
        assert mode_1['frequency'] == -1178.9
        assert mode_1['intensity'] == 112.0
    
    def test_limited_freq(self):

        parameter_config = deepcopy(parameters)
        parameter_config['DEMON_PARAMETERS']['ACTIVE'].update(
            {"FREQ":10,}
        )
        dem = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=".run/freq/",
            **parameter_config
        )
        dem.clean_workdir()

        dem.calculate(
            symbols=image.symbols,
            positions=image.positions
        )

        results = dem.results
        assert results["zpe"] == 0.0475
        assert len(results["frequency"]) == 16
        mode_1 = results["frequency"][0]
        assert mode_1['mode'] == 1
        assert mode_1['frequency'] == -1178.9
        assert mode_1['intensity'] == 112.0


    def test_freq_ci(self):


        parameter_config = deepcopy(parameters)
        parameter_config['DEMON_PARAMETERS']['ACTIVE'].update(
            {
                "FREQ":True,
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
            workdir=".run/freq/",
            **parameter_config
        )
        dem.clean_workdir()
        freq_file = os.path.join(dem.workdir,'deMon.freq')


        dem.calculate(
            symbols=image.symbols,
            positions=image.positions
        )

        results = dem.results
        assert results["zpe"] == 0.0477
        assert len(results["frequency"]) == 18
        mode_1 = results["frequency"][0]
        assert mode_1['mode'] == 1
        assert mode_1['frequency'] == -1134.2
        assert mode_1['intensity'] == 79.6


class TestTDDFTB:

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






class TestDFTBCI:

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






        