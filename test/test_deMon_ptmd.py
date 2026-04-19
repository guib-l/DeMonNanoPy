import os

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

WORKDIR = ".run/ptmd/"




class TestPTMD:

    def test_md_parallel(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE":{
                    "ACTIVE":{
                    "MD":{
                        "MDYNAMICS":{
                            "RANDOM":200,
                        },
                        "TIMESTEP":0.4,
                        "MDSTEP":{
                            "MAX":548,
                            "OUT":1,
                            "SOUT":1,
                        },
                        "TRAJECTORY":True,
                        "PARATEMP":{
                            "TMINIMUM":100,
                            "TMAXIMUM":300,
                            "NUMREP":3,
                            "NBREQ":100,
                            "EXSTEP":10,
                            "SHORT":False,
                            "LINEAR":False
                        },
                    },
                }
            }
        })

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

        msg = 'ERROR IN MPI_INIT'
        if msg in results["errors"][0]['message']:
            return

        pote = results["potential_energy"]
        kine = results["kinetic_energy"]
        tote = results["total_energy"]

        diff = np.sum((tote - (pote+kine))[1:])
        assert diff <= 1e-5, "Energy conserved"

    def test_md_numbRep(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE":{
                    "ACTIVE":{
                    "MD":{
                        "MDYNAMICS":{
                            "RANDOM":200,
                        },
                        "TIMESTEP":0.4,
                        "MDSTEP":{
                            "MAX":548,
                            "OUT":1,
                            "SOUT":1,
                        },
                        "TRAJECTORY":True,
                        "PARATEMP":{
                            "TMINIMUM":100,
                            "TMAXIMUM":300,
                            "NUMREP":4,
                            "NBREQ":100,
                            "EXSTEP":10,
                            "SHORT":True,
                            "LINEAR":False
                        },
                    },
                }
            }
        })

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
        msg = 'ERROR IN MPI_INIT'
        if msg in results["errors"][0]['message']:
            return
        
        pote = results["potential_energy"]
        kine = results["kinetic_energy"]
        tote = results["total_energy"]

        diff = np.sum((tote - (pote+kine))[1:])
        assert diff <= 1e-5, "Energy conserved"



    def test_md_short(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE":{
                    "ACTIVE":{
                    "MD":{
                        "MDYNAMICS":{
                            "RANDOM":200,
                        },
                        "TIMESTEP":0.4,
                        "MDSTEP":{
                            "MAX":548,
                            "OUT":1,
                            "SOUT":1,
                        },
                        "TRAJECTORY":True,
                        "PARATEMP":{
                            "TMINIMUM":100,
                            "TMAXIMUM":300,
                            "NUMREP":3,
                            "NBREQ":100,
                            "EXSTEP":10,
                            "SHORT":True,
                            "LINEAR":False
                        },
                    },
                }
            }
        })

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
        msg = 'ERROR IN MPI_INIT'
        if msg in results["errors"][0]['message']:
            return
        pote = results["potential_energy"]
        kine = results["kinetic_energy"]
        tote = results["total_energy"]

        diff = np.sum((tote - (pote+kine))[1:])
        assert diff <= 1e-5, "Energy conserved"



    def test_md_linear(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE":{
                    "ACTIVE":{
                    "MD":{
                        "MDYNAMICS":{
                            "RANDOM":200,
                        },
                        "TIMESTEP":0.4,
                        "MDSTEP":{
                            "MAX":548,
                            "OUT":1,
                            "SOUT":1,
                        },
                        "TRAJECTORY":True,
                        "PARATEMP":{
                            "TMINIMUM":100,
                            "TMAXIMUM":300,
                            "NUMREP":3,
                            "NBREQ":100,
                            "EXSTEP":10,
                            "SHORT":False,
                            "LINEAR":True
                        },
                    },
                }
            }
        })

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
        msg = 'ERROR IN MPI_INIT'
        if msg in results["errors"][0]['message']:
            return
        
        pote = results["potential_energy"]
        kine = results["kinetic_energy"]
        tote = results["total_energy"]

        diff = np.sum((tote - (pote+kine))[1:])
        assert diff <= 1e-5, "Energy conserved"















