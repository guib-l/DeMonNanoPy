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
                "SCC":True,
                "DISP":2
            },
        },
    }
}

image = Atoms(
    ["O","H","H","C","H"] + ["C"]*5 + ["H"]*5,
    positions=np.array([
            [ 0.346614,-0.028767, 3.138366],
            [ 0.445000,-0.234326, 2.203712],
            [-0.256186, 0.712141, 3.220164],
            [ 0.034713, 1.430978,-0.175630],
            [ 0.031596, 2.526523,-0.271684],
            [ 1.245204, 0.738039,-0.136167],
            [ 1.249216,-0.652897,-0.021609],
            [ 0.042421,-1.350605, 0.052487],
            [-1.167840,-0.656928, 0.012832],
            [-1.171586, 0.733714,-0.101242],
            [ 2.195868, 1.287371,-0.197495],
            [ 2.203124,-1.199372, 0.007038],
            [ 0.045572,-2.446540, 0.140211],
            [-2.118619,-1.207078, 0.067882],
            [-2.125716, 1.279602,-0.138796],
        ])
    ) 

WORKDIR = ".run/print/"




class TestDftbBasis:


    def _test_scc_moe(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "PRINT":{
                    "MOE":True
                }
            }
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
        
        assert np.allclose(results["energy"]["energy"],-16.64681888, atol=1e-7)

    def test_scc_mos(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "PRINT":{
                    "MOS":True
                }
            }
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
        
        assert np.allclose(results["energy"]["energy"],-16.64681888, atol=1e-7)


    def test_scc_de2(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "PRINT":{
                    "DE2":True,
                }
            }
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
        assert np.allclose(results["energy"]["energy"],-16.64681888, atol=1e-7)

    def test_scc_mc(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "PRINT":{
                    "MC":True
                }
            }
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
        assert np.allclose(results["energy"]["energy"],-16.64681888, atol=1e-7)

    def test_scc_ase(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "PRINT":{
                    "ASE":True,
                }
            }
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
        assert np.allclose(results["energy"]["energy"],-16.64681888, atol=1e-7)


    def test_scc_debug(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "PRINT":{
                    "DEBUG":True
                }
            }
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **copy_parameters
        )
            
        mod.calculate(
            symbols=image.symbols,
            positions=image.positions,
            extract_debug=True,
            keep_matrix="all"
        )

        results = mod.results

        assert np.allclose(results["energy"]["energy"],-16.64681888, atol=1e-7)

        assert np.allclose(np.shape(results["Gamma"]),(15,15), atol=1e-7)
        assert np.allclose(np.shape(results["Overlaps"][-1]),(36,36), atol=1e-7)
        assert np.allclose(np.shape(results["Ham0"][-1]),(36,36), atol=1e-7)
        assert np.allclose(np.shape(results["Hamiltonian"][-1]),(36,36), atol=1e-7)
        assert np.allclose(np.shape(results["Density"][-1]),(36,36), atol=1e-7)
        assert np.allclose(np.shape(results["Coefficients"][-1]),(36,36), atol=1e-7)






