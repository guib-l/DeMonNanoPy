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
    positions=np.array(
        [[1.2478,-0.5185,3.4049],
        [1.5946,-1.4204,3.3886],
        [0.9008,-0.3341,2.5062],
        [3.2478,-0.4185,3.4049],
        [3.5946,-1.5204,3.3886],
        [2.9008,-0.3341,2.6062],
        ])
    )

WORKDIR = ".run/opt/"





class TestOptimization:

    def test_opt_basic(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE":{
                    "ACTIVE":{
                        "OPT":{
                            "MAX":999,
                            "OUT":1,
                            "TRAJECTORY":True
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
        assert results["energy"]["energy"] == -8.15563755       


    def test_opt_maxiter(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE":{
                    "ACTIVE":{
                        "OPT":{
                            "MAX":999,
                            "TOL":3e-4,
                            "GRADTOL":4e-6,
                            "STEP":0.3,
                            "CGRAD":True,
                            "SDC":False,
                            "OUT":1,
                            "TRAJECTORY":True
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
        assert results["energy"]["energy"] == -8.1487639     


    def test_opt_maxiter(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE":{
                    "ACTIVE":{
                        "OPT":{
                            "MAX":10,
                            "TOL":3e-4,
                            "GRADTOL":4e-6,
                            "STEP":0.3,
                            "CGRAD":True,
                            "SDC":False,
                            "OUT":1,
                            "TRAJECTORY":True
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
        assert results["energy"]["energy"] == -8.1487639 
        assert len(results["trajectory"]) == 11

    def test_opt_tolerance(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE":{
                    "ACTIVE":{
                        "OPT":{
                            "MAX":999,
                            "TOL":3e-2,
                            "GRADTOL":4e-6,
                            "STEP":0.3,
                            "CGRAD":True,
                            "SDC":False,
                            "OUT":1,
                            "TRAJECTORY":True
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
        assert results["energy"]["energy"] == -8.15563755   

    def test_opt_gradtol(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE":{
                    "ACTIVE":{
                        "OPT":{
                            "MAX":99,
                            "TOL":3e-4,
                            "GRADTOL":4e-3,
                            "STEP":0.3,
                            "CGRAD":True,
                            "SDC":False,
                            "OUT":1,
                            "TRAJECTORY":True
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
        assert results["energy"]["energy"] == -8.15146881

    def test_opt_cgrad(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE":{
                    "ACTIVE":{
                        "OPT":{
                            "MAX":99,
                            "TOL":3e-4,
                            "GRADTOL":4e-6,
                            "STEP":0.3,
                            "CGRAD":True,
                            "SDC":False,
                            "OUT":1,
                            "TRAJECTORY":True
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
        assert results["energy"]["energy"] == -8.1553


    def test_opt_steepest_default(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE":{
                    "ACTIVE":{
                        "OPT":{
                            "MAX":99,
                            "TOL":3e-4,
                            "GRADTOL":4e-6,
                            "STEP":0.3,
                            "CGRAD":False,
                            "SDC":True,
                            "OUT":1,
                            "TRAJECTORY":True
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
        assert results["energy"]["energy"] == -8.14859822   


    def test_opt_steepest(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE":{
                    "ACTIVE":{
                        "OPT":{
                            "MAX":99,
                            "TOL":3e-4,
                            "GRADTOL":4e-6,
                            "STEP":0.3,
                            "CGRAD":False,
                            "SDC":0.2,
                            "OUT":1,
                            "TRAJECTORY":True
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
        assert results["energy"]["energy"] == -8.14858038   


    def test_opt_out(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE":{
                    "ACTIVE":{
                        "OPT":{
                            "OUT":10,
                            "MAX":20,
                            "TOL":3e-4,
                            "GRADTOL":4e-6,
                            "STEP":0.3,
                            "CGRAD":True,
                            "SDC":False,
                            "NUCFRIC":False,
                            "TRAJECTORY":True
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
        assert results["energy"]["energy"] == -8.14955388  
        assert len(results["trajectory"]) == 3
        #print(" > FAILED IN PRACTICE")



    def test_opt_noTraj(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE":{
                    "ACTIVE":{
                        "OPT":{
                            "MAX":99,
                            "TOL":3e-4,
                            "GRADTOL":4e-6,
                            "STEP":0.3,
                            "CGRAD":True,
                            "SDC":False,
                            "OUT":10,
                            "TRAJECTORY":False
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
        assert results["energy"]["energy"] == -8.1553
        assert "trajectory" not in results.keys() 






