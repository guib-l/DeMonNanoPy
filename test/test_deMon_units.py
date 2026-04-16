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
            "GEOMETRY":{
                "BOHR":False,
                "ANGST":False,
                "CARTE":False
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

WORKDIR = ".run/units/"


class TestUnits:

    def test_units_bohr(self):

        BOHR = 0.529177
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters['DEMON_PARAMETERS']['ACTIVE'].update(
            {"GEOMETRY":{"BOHR":True},}
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **copy_parameters
        )

        mod.calculate(
            symbols=image.symbols,
            positions=image.positions / BOHR
        )

        results = mod.results
        assert np.allclose(results["energy"]["energy"], -8.06209368, atol=1e-7)


    def test_units_angst(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters['DEMON_PARAMETERS']['ACTIVE'].update(
            {"GEOMETRY":{"ANGST":True},}
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **copy_parameters
        )

        mod.calculate(
            symbols=image.symbols,
            positions=image.positions
        )

        results = mod.results
        assert np.allclose(results["energy"]["energy"],-8.06209343 , atol=1e-7)


    def test_units_carte(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters['DEMON_PARAMETERS']['ACTIVE'].update(
            {"GEOMETRY":{"CARTE":True},}
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **copy_parameters
        )

        mod.calculate(
            symbols=image.symbols,
            positions=image.positions
        )

        results = mod.results
        assert np.allclose(results["energy"]["energy"],-8.06209343 , atol=1e-7)
