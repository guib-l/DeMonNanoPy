import sys
import os

import pytest
#import configs
import numpy as np

import copy
from copy import deepcopy

from ase.atoms import Atoms
from deMonPy.deMonNano import deMonNano
from deMonPy.deMonNano import Module_DeMonNano


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
            "QUATERNION":{
                "BOHR":False,
                "ANGST":False,
                'COORDS':None,
                "RIGID":False
            },
            "MOLECULES":{
                "NAMES":[],
            }
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

WORKDIR = ".run/molecule/"


class TestMolecule:

    def test_molecule(self):
        import shutil
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters['DEMON_PARAMETERS']['ACTIVE'].update(
            {"MOLECULES":{"NAMES":["WAT","BZZ","WAT"]},
             "QUATERNION":{
                 'COORDS':np.array(
                     [
                         [1.0, 0.0, .0  ,1.0 ,0.0, 0.0, 0.0],
                         [0. ,-3. , 0.0 ,0.7 ,0.7, 0. , 0.0],
                         [1. ,-3. ,-4.0 ,1.0 ,0.0, 0.0, 0.0 ]
                     ]
                 )
             }}
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **copy_parameters
        )

        shutil.copy2("./data/MOLECULES", f"{WORKDIR}/MOLECULES")

        mod.calculate(
            symbols=image.symbols,
            positions=image.positions,
            clean_repository=False
        )

        results = mod.results
        assert np.allclose(results["energy"]["energy"], -20.7065151, atol=1e-7)


    def test_molecule_angst(self):
        import shutil
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters['DEMON_PARAMETERS']['ACTIVE'].update(
            {"MOLECULES":{"NAMES":["WAT","BZZ","WAT"]},
             "QUATERNION":{
                 "ANGST":True,
                 'COORDS':np.array(
                     [
                         [1.0, 0.0, .0  ,1.0 ,0.0, 0.0, 0.0],
                         [0. ,-3. , 0.0 ,0.7 ,0.7, 0. , 0.0],
                         [1. ,-3. ,-4.0 ,1.0 ,0.0, 0.0, 0.0 ]
                     ]
                 )
             }}
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **copy_parameters
        )

        shutil.copy2("./data/MOLECULES", f"{WORKDIR}/MOLECULES")

        mod.calculate(
            symbols=image.symbols,
            positions=image.positions,
            clean_repository=False
        )

        results = mod.results
        assert np.allclose(results["energy"]["energy"], -20.7065151, atol=1e-7)

    def test_molecule_bohr(self):
        import shutil
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters['DEMON_PARAMETERS']['ACTIVE'].update(
            {"MOLECULES":{"NAMES":["WAT","BZZ","WAT"]},
             "QUATERNION":{
                 "BOHR":True,
                 'COORDS':np.array(
                     [
                         [1.0, 0.0, .0  ,1.0 ,0.0, 0.0, 0.0],
                         [0. ,-3. , 0.0 ,0.7 ,0.7, 0. , 0.0],
                         [1. ,-3. ,-4.0 ,1.0 ,0.0, 0.0, 0.0 ]
                     ]
                 )
             }}
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **copy_parameters
        )

        shutil.copy2("./data/MOLECULES", f"{WORKDIR}/MOLECULES")

        mod.calculate(
            symbols=image.symbols,
            positions=image.positions,
            clean_repository=False
        )

        results = mod.results
        assert np.allclose(results["energy"]["energy"], -1.06599675, atol=1e-7)


    @pytest.mark.beta
    def test_molecule_rigid(self):
        import shutil
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters['DEMON_PARAMETERS']['ACTIVE'].update(
            {"MOLECULES":{"NAMES":["WAT","BZZ","WAT"]},
             "QUATERNION":{
                 "RIGID":True,
                 'COORDS':np.array(
                     [
                         [1.0, 0.0, .0  ,1.0 ,0.0, 0.0, 0.0],
                         [0. ,-3. , 0.0 ,0.7 ,0.7, 0. , 0.0],
                         [1. ,-3. ,-4.0 ,1.0 ,0.0, 0.0, 0.0 ]
                     ]
                 )
             }}
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **copy_parameters
        )

        shutil.copy2("./data/MOLECULES", f"{WORKDIR}/MOLECULES")

        mod.calculate(
            symbols=image.symbols,
            positions=image.positions,
            clean_repository=False
        )

        results = mod.results
        assert np.allclose(results["energy"]["energy"], -20.7065151, atol=1e-7)  










