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

WORKDIR = ".run/md/"




class TestMDbasic:

    def test_md_basic(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE":{
                    "ACTIVE":{
                    "MD":{
                        "MDYNAMICS":{
                            "ZERO":True,
                            "RANDOM":300,
                            "RAN":True,
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
                            "MAX":1500,
                            "OUT":1,
                            "SOUT":1,
                            "TSIM":None
                        },
                        "MDTEMP":300,
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

        assert np.sum((tote - (pote+kine))[1:]) <= 1e-5
        



    def test_md_start_zero(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE":{
                    "ACTIVE":{
                    "MD":{
                        "MDYNAMICS":{
                            "ZERO":True,
                            "RANDOM":False,
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
                            "MAX":1500,
                            "OUT":1,
                            "SOUT":1,
                            "TSIM":None
                        },
                        "MDTEMP":None,
                        "TRAJECTORY":True 
                        },
                    },
                }
            }
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **copy_parameters)

        mod.calculate(
            symbols=image.symbols,
            positions=image.positions)

        results = mod.results
        pote = results["potential_energy"]
        kine = results["kinetic_energy"]
        tote = results["total_energy"]

        assert np.sum((tote - (pote+kine))[1:]) <= 1e-5
        traj = results["trajectory"]
        temperature = [atm.get_temperature() for atm in traj]

        assert temperature[0]<0.001 and temperature[0]>=0.00000


    def test_md_start_155(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE":{
                    "ACTIVE":{
                    "MD":{
                        "MDYNAMICS":{
                            "ZERO":False,
                            "RANDOM":155,
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
                            "MAX":100,
                            "OUT":1,
                            "SOUT":1,
                            "TSIM":None
                        },
                        "MDTEMP":None,
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
        
        assert np.sum((tote - (pote+kine))[1:]) <= 1e-5
        
        traj = results["trajectory"]
        temperature = [atm.get_temperature() for atm in traj]

        assert temperature[0]<155.1 and temperature[0]>154.9

        velocity = traj[0].get_velocities()
        #print( traj[0].get_angular_momentum( ) )


    def test_md_start_ran(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE":{
                    "ACTIVE":{
                    "MD":{
                        "MDYNAMICS":{
                            "ZERO":False,
                            "RANDOM":None,
                            "RAN":True,
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
                            "MAX":100,
                            "OUT":1,
                            "SOUT":1,
                            "TSIM":None
                        },
                        "MDTEMP":224,
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
        
        assert np.sum((tote - (pote+kine))[1:]) <= 1e-5
        
        traj = results["trajectory"]
        temperature = [atm.get_temperature() for atm in traj]

        diff = np.abs(temperature[0] - 224)
        assert diff < 0.1, "Starting temperature of RAN parameters is not tacken into account"

        velocity = traj[0].get_velocities()
        #print( traj[0].get_angular_momentum( ) )




    def test_md_velocity(self):

        import ase
        from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary,ZeroRotation)
            
        MaxwellBoltzmannDistribution(atoms=image, 
                                    temperature_K=482, 
                                    force_temp=True,
                                    rng=np.random.RandomState(314159262))
        velocities = image.get_velocities() * ase.units.fs

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE":{
                    "ACTIVE":{
                    "MD":{
                        "MDYNAMICS":{
                            "ZERO":False,
                            "RANDOM":None,
                            "RAN":False,
                            "READ":{
                                "VELOCITIES":velocities
                            },
                            "RESET":False,
                            "WALL":None,
                            "EXP":None,
                            "ENER":None
                        },
                        "TIMESTEP":0.4,
                        "MDSTEP":{
                            "MAX":100,
                            "OUT":1,
                            "SOUT":1,
                            "TSIM":None
                        },
                        "MDTEMP":224,
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
        
        assert np.sum((tote - (pote+kine))[1:]) <= 1e-5
        
        traj = results["trajectory"]
        temperature = [atm.get_temperature() for atm in traj]

        assert temperature[0]<482.1 and temperature[0]>481.9

        velocity = traj[0].get_velocities()




    def test_md_wall_cart(self):

        import ase
        from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary,ZeroRotation)
            
        MaxwellBoltzmannDistribution(atoms=image, 
                                    temperature_K=482, 
                                    force_temp=True,
                                    rng=np.random.RandomState(314159262))
        velocities = image.get_velocities() * ase.units.fs

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE":{
                    "ACTIVE":{
                    "MD":{
                        "MDYNAMICS":{
                            "ZERO":False,
                            "RANDOM":300,
                            "RAN":False,
                            "READ":{
                                "VELOCITIES":[]
                            },
                            "RESET":False,
                            "WALL":12.,
                            "EXP":0.1,
                            "ENER":1.0
                        },
                        "TIMESTEP":0.4,
                        "MDSTEP":{
                            "MAX":1000,
                            "OUT":1,
                            "SOUT":1,
                            "TSIM":None
                        },
                        "MDTEMP":224,
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
        
        assert np.sum((tote - (pote+kine))[1:]) <= 1e-5
        
        traj = results["trajectory"]
        temperature = [atm.get_temperature() for atm in traj]

        assert temperature[0]<224.1 and temperature[0]>223.9


    def test_md_reset(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE":{
                    "ACTIVE":{
                    "MD":{
                        "MDYNAMICS":{
                            "ZERO":True,
                            "RANDOM":300,
                            "RAN":True,
                            "READ":{
                                "VELOCITIES":[]
                            },
                            "RESET":True,
                            "WALL":None,
                            "EXP":None,
                            "ENER":None
                        },
                        "TIMESTEP":0.4,
                        "MDSTEP":{
                            "MAX":1500,
                            "OUT":1,
                            "SOUT":1,
                            "TSIM":None
                        },
                        "MDTEMP":300,
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

        assert np.sum((tote - (pote+kine))[1:]) <= 1e-5





