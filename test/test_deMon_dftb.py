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



class TestDftbBasis:

    def test_basic(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS":{
                    "PTYPE":"NSC",
                    "SKFILE":deMonPy.DEMON_BASIS
                },
                "DEMON_PARAMETERS":{
                    "ACTIVE":{
                        "DFTB":{
                            "SCC":False,
                        },
                    },
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
        assert results["energy"]["energy"] == -8.12781231       

    def test_scc_bio(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS":{
                    "PTYPE":"BIO",
                    "SKFILE":deMonPy.DEMON_BASIS
                },
                "DEMON_PARAMETERS":{
                    "ACTIVE":{
                        "DFTB":{
                            "SCC":True,
                        },
                    },
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
        assert results["energy"]["energy"] == -8.06209343 

    def test_scc_mat(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS":{
                    "PTYPE":"MAT",
                    "SKFILE":deMonPy.DEMON_BASIS
                },
                "DEMON_PARAMETERS":{
                    "ACTIVE":{
                        "DFTB":{
                            "SCC":True,
                        },
                    },
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
        assert results["energy"]["energy"] == -8.08755742 


WORKDIR = ".run/dftb/"

class TestDftb:

    def test_scc(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS":{
                    "ACTIVE":{
                        "DFTB":{
                            "SCC":True,
                            "TOL":1e-8,
                            "MEMOSCC":False,
                            "POLA":False,
                            "MAX":100,
                            "MIX":0.2,
                            "SIMPLE":False,
                            "L-DEP":False,
                            "FERMI":None,
                            "THRID":False,
                            "DISP":False,
                            "LEV_S":None
                        },
                    },
                }
            }
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **copy_parameters )
            
        mod.calculate(
            symbols=image.symbols,
            positions=image.positions )

        results = mod.results
        assert results["energy"]["energy"] == -8.06209343 

    def test_scc_tol(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS":{
                    "ACTIVE":{
                        "DFTB":{
                            "SCC":True,
                            "TOL":1e-2,
                            "MEMOSCC":False,
                            "POLA":False,
                            "MAX":100,
                            "MIX":0.2,
                            "SIMPLE":False,
                            "L-DEP":False,
                            "FERMI":None,
                            "THRID":False,
                            "DISP":False,
                            "LEV_S":None
                        },
                    },
                }
            }
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **copy_parameters )
            
        mod.calculate(
            symbols=image.symbols,
            positions=image.positions )

        results = mod.results
        assert results["energy"]["energy"] == -8.06203571 


    def test_scc_memoscc(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS":{
                    "ACTIVE":{
                        "DFTB":{
                            "SCC":True,
                            "TOL":1e-8,
                            "MEMOSCC":True,
                            "POLA":False,
                            "MAX":100,
                            "MIX":0.2,
                            "SIMPLE":False,
                            "L-DEP":False,
                            "FERMI":None,
                            "THRID":False,
                            "DISP":False,
                            "LEV_S":None
                        },
                    },
                }
            }
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **copy_parameters )
            
        mod.calculate(
            symbols=image.symbols,
            positions=image.positions )

        results = mod.results
        assert results["energy"]["energy"] == -8.06209343 


    def test_scc_pola(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS":{
                    "ACTIVE":{
                        "DFTB":{
                            "SCC":True,
                            "TOL":1e-8,
                            "MEMOSCC":False,
                            "POLA":True,
                            "MAX":100,
                            "MIX":0.2,
                            "SIMPLE":False,
                            "L-DEP":False,
                            "FERMI":None,
                            "THRID":False,
                            "DISP":False,
                            "LEV_S":None
                        },
                    },
                }
            }
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **copy_parameters )
            
        mod.calculate(
            symbols=image.symbols,
            positions=image.positions )

        results = mod.results
        assert results["energy"]["energy"] == -7.92780227 

    def test_scc_maxiter(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS":{
                    "ACTIVE":{
                        "DFTB":{
                            "SCC":True,
                            "TOL":1e-8,
                            "MEMOSCC":False,
                            "POLA":False,
                            "MAX":5,
                            "MIX":0.2,
                            "SIMPLE":False,
                            "L-DEP":False,
                            "FERMI":None,
                            "THRID":False,
                            "DISP":False,
                            "LEV_S":None
                        },
                    },
                }
            }
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **copy_parameters )
            
        mod.calculate(
            symbols=image.symbols,
            positions=image.positions )

        results = mod.results
        assert "loop over charges not converged" in results["errors"] 
        

    def test_scc_mixing(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS":{
                    "ACTIVE":{
                        "DFTB":{
                            "SCC":True,
                            "TOL":1e-8,
                            "MEMOSCC":False,
                            "POLA":False,
                            "MAX":100,
                            "MIX":0.02,
                            "SIMPLE":False,
                            "L-DEP":False,
                            "FERMI":None,
                            "THRID":False,
                            "DISP":False,
                            "LEV_S":None
                        },
                    },
                }
            }
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **copy_parameters )
            
        mod.calculate(
            symbols=image.symbols,
            positions=image.positions )

        results = mod.results
        assert results["energy"]["energy"] == -8.06209343 

    @pytest.mark.xfail(reason="NOT CRITICAL -> TO FIX")
    def test_scc_simple_mixing(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS":{
                    "ACTIVE":{
                        "DFTB":{
                            "SCC":True,
                            "TOL":1e-8,
                            "MEMOSCC":False,
                            "POLA":False,
                            "MAX":100,
                            "MIX":0.2,
                            "SIMPLE":True,
                            "L-DEP":False,
                            "FERMI":None,
                            "THRID":False,
                            "DISP":False,
                            "LEV_S":None
                        },
                    },
                }
            }
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **copy_parameters )
            
        mod.calculate(
            symbols=image.symbols,
            positions=image.positions )

        results = mod.results 
        assert results["energy"]["energy"] == -8.00903688 

    @pytest.mark.xfail(reason="NOT CRITICAL -> TO FIX")
    def test_scc_ldep(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS":{
                    "ACTIVE":{
                        "DFTB":{
                            "SCC":True,
                            "TOL":1e-8,
                            "MEMOSCC":False,
                            "POLA":False,
                            "MAX":100,
                            "MIX":0.2,
                            "SIMPLE":False,
                            "L-DEP":True,
                            "FERMI":None,
                            "THRID":False,
                            "DISP":False,
                            "LEV_S":None
                        },
                    },
                }
            }
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **copy_parameters )
        
        mod.clean_workdir()
            
        mod.calculate(
            symbols=image.symbols,
            positions=image.positions )

        results = mod.results
        assert results["energy"]["energy"] == -8.00903688 

    def test_scc_fermi(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS":{
                    "ACTIVE":{
                        "DFTB":{
                            "SCC":True,
                            "TOL":1e-8,
                            "MEMOSCC":False,
                            "POLA":False,
                            "MAX":100,
                            "MIX":0.2,
                            "SIMPLE":False,
                            "L-DEP":False,
                            "FERMI":50.00,
                            "THRID":False,
                            "DISP":False,
                            "LEV_S":None
                        },
                    },
                }
            }
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **copy_parameters )
            
        mod.calculate(
            symbols=image.symbols,
            positions=image.positions )

        results = mod.results
        assert results["energy"]["energy"] == -8.06209343 




    def test_scc_disp1(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS":{
                    "ACTIVE":{
                        "DFTB":{
                            "SCC":True,
                            "TOL":1e-8,
                            "MEMOSCC":False,
                            "POLA":False,
                            "MAX":100,
                            "MIX":0.2,
                            "SIMPLE":False,
                            "L-DEP":False,
                            "FERMI":None,
                            "THRID":False,
                            "DISP":1,
                            "LEV_S":None
                        },
                    },
                }
            }
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **copy_parameters )
            
        mod.calculate(
            symbols=image.symbols,
            positions=image.positions )

        results = mod.results
        assert results["energy"]["energy"] == -8.04913036 

    def test_scc_disp2(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS":{
                    "ACTIVE":{
                        "DFTB":{
                            "SCC":True,
                            "TOL":1e-8,
                            "MEMOSCC":False,
                            "POLA":False,
                            "MAX":100,
                            "MIX":0.2,
                            "SIMPLE":False,
                            "L-DEP":False,
                            "FERMI":None,
                            "THRID":False,
                            "DISP":2,
                            "LEV_S":None
                        },
                    },
                }
            }
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **copy_parameters )
            
        mod.calculate(
            symbols=image.symbols,
            positions=image.positions )

        results = mod.results
        assert results["energy"]["energy"] == -8.06209886 

    def test_scc_level_shift(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS":{
                    "ACTIVE":{
                        "DFTB":{
                            "SCC":True,
                            "TOL":1e-8,
                            "MEMOSCC":False,
                            "POLA":False,
                            "MAX":100,
                            "MIX":0.2,
                            "SIMPLE":False,
                            "L-DEP":False,
                            "FERMI":None,
                            "THRID":False,
                            "DISP":False,
                            "LEV_S":0.0001
                        },
                    },
                }
            }
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **copy_parameters )
            
        mod.calculate(
            symbols=image.symbols,
            positions=image.positions )

        results = mod.results


        assert np.allclose(
            results["energy"]["energy"], 
            -8.06209343, 
            atol=1e-7
        )

        




