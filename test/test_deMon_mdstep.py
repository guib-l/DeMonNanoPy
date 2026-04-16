import os

import numpy as np

import copy
from copy import deepcopy

import pytest
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

WORKDIR = ".run/mdstep/"




class TestMDstep:

    def test_md_maxStep(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE":{
                    "ACTIVE":{
                    "MD":{
                        "MDYNAMICS":{
                            "ZERO":True,
                            "RANDOM":300,
                            "RAN":False,
                            "READ":{
                                "VELOCITIES":[]
                            },
                            "RESET":False,
                            "WALL":None,
                            "EXP":None,
                            "ENER":None
                        },
                        "TIMESTEP":0.4,
                        "MDSTEP":{
                            "MAX":548,
                            "OUT":1,
                            "SOUT":1,
                            "TSIM":None
                        },
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
        pote = results["potential_energy"]
        kine = results["kinetic_energy"]
        tote = results["total_energy"]

        diff = np.sum((tote - (pote+kine))[1:])
        assert diff <= 1e-5, "Energy conserved"

        traj = results["trajectory"]
        assert len(traj) == 549, "MAX criteria is not conserved"

        temperature = [atm.get_temperature() for atm in traj]
        diff = np.abs(temperature[0] - 300)
        assert diff < 0.1, "Starting temperature is not tacken into account"


    def test_md_outStep(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE":{
                    "ACTIVE":{
                    "MD":{
                        "MDYNAMICS":{
                            "ZERO":True,
                            "RANDOM":300,
                            "RAN":False,
                            "READ":{
                                "VELOCITIES":[]
                            },
                            "RESET":False,
                            "WALL":None,
                            "EXP":None,
                            "ENER":None
                        },
                        "TIMESTEP":0.4,
                        "MDSTEP":{
                            "MAX":548,
                            "OUT":10,
                            "SOUT":1,
                            "TSIM":None
                        },
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
        pote = results["potential_energy"]
        kine = results["kinetic_energy"]
        tote = results["total_energy"]

        diff = np.sum((tote - (pote+kine))[1:])
        assert diff <= 1e-5, "Energy conserved"

        traj = results["trajectory"]
        assert len(traj) == 56, "OUT criteria is not conserved"

        temperature = [atm.get_temperature() for atm in traj]
        diff = np.abs(temperature[0] - 300)
        assert diff < 0.1, "Starting temperature is not tacken into account"


    def test_md_soutStep(self):
        # TODO: get the number of lines in outputs
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE":{
                    "ACTIVE":{
                    "MD":{
                        "MDYNAMICS":{
                            "ZERO":True,
                            "RANDOM":300,
                            "RAN":False,
                            "READ":{
                                "VELOCITIES":[]
                            },
                            "RESET":False,
                            "WALL":None,
                            "EXP":None,
                            "ENER":None
                        },
                        "TIMESTEP":0.4,
                        "MDSTEP":{
                            "MAX":105,
                            "OUT":1,
                            "SOUT":10,
                            "TSIM":None
                        },
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
        pote = results["potential_energy"]
        kine = results["kinetic_energy"]
        tote = results["total_energy"]

        diff = np.sum((tote - (pote+kine))[1:])
        assert diff <= 1e-5, "Energy conserved"

        traj = results["trajectory"]
        assert len(traj) == 106, "OUT criteria is not conserved"

        temperature = [atm.get_temperature() for atm in traj]
        diff = np.abs(temperature[0] - 300)
        assert diff < 0.1, "Starting temperature is not tacken into account"


    @pytest.mark.xfail(reason="NOT CRITICAL -> TO FIX")
    def test_md_tsimStep(self):
        # TODO: get the number of lines in outputs
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE":{
                    "ACTIVE":{
                    "MD":{
                        "MDYNAMICS":{
                            "ZERO":True,
                            "RANDOM":300,
                            "RAN":False,
                            "READ":{
                                "VELOCITIES":[]
                            },
                            "RESET":False,
                            "WALL":None,
                            "EXP":None,
                            "ENER":None
                        },
                        "TIMESTEP":0.4,
                        "MDSTEP":{
                            "MAX":99999,
                            "OUT":1,
                            "SOUT":1,
                            "TSIM":0.54
                        },
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
        mod.clean_workdir()

        mod.calculate(
            symbols=image.symbols,
            positions=image.positions
        )

        results = mod.results
        pote = results["potential_energy"]
        kine = results["kinetic_energy"]
        tote = results["total_energy"]

        diff = np.sum((tote - (pote+kine))[1:])
        assert diff <= 1e-5, "Energy conserved"

        traj = results["trajectory"]
        temperature = [atm.get_temperature() for atm in traj]
        diff = np.abs(temperature[0] - 300)
        assert diff < 0.1, "Starting temperature is not tacken into account"






