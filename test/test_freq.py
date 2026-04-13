
import os
import sys

import numpy as np

from copy import deepcopy

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

WORKDIR = ".run/freq/"

class TestFreq:

    def test_freq(self):

        parameter_config = deepcopy(parameters)
        parameter_config['DEMON_PARAMETERS']['ACTIVE'].update(
            {"FREQ":True,}
        )
        dem = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
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
            {"FREQ":{"VIB":5},}
        )
        dem = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **parameter_config
        )
        dem.clean_workdir()

        dem.calculate(
            symbols=image.symbols,
            positions=image.positions
        )

        results = dem.results
        assert results["zpe"] == 0.0475
        assert len(results["frequency"]) == 11
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
            workdir=WORKDIR,
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





        