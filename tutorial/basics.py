import os
import numpy as np

import copy
from copy import deepcopy

from ase.atoms import Atoms

import deMonPy
from deMonPy.deMonNano import deMonNano

# [MANDATORY]

# Where the calculation is done
WORKDIR = ".run/opt/"

deMonPy.configure_from_file(os.path.join("..", "global.json"))


def MyFirstCalculation():

    # Minimal parameters
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
                },
            },
        }
    }
    
    # Used ase.Atoms
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
    
    import ase
    from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                     Stationary,ZeroRotation)
    
    MaxwellBoltzmannDistribution(atoms=image, 
                                 temperature_K=482, 
                                 force_temp=True,
                                 rng=np.random.RandomState(314159262))
    velocities = image.get_velocities() * ase.units.fs
    
    parameters.update(
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
                                "WALL":None,
                                "EXP":None,
                                "ENER":None,
                            },
                            "TIMESTEP":0.4,
                            "MDSTEP":{
                                "MAX":500,
                                "OUT":1,
                                "SOUT":1,
                                "TSIM":None
                            },
                            "MDTEMP":300   ,
                            "TRAJECTORY":True
                            },
                    },
                }
            }
        )


    dem = deMonNano(
        title="CALCULATION DEMONANO",
        workdir=WORKDIR,
        **parameters
    )

    dem.calculate(
        symbols=image.symbols,
        positions=image.positions
    )

    dem.print_results()

    results = dem.results
    
    pote = results["potential_energy"]
    kine = results["kinetic_energy"]
    tote = results["total_energy"]

    print( np.sum((tote - (pote+kine))[1:]) )

    traj = results["trajectory"]

    temperature = [atm.get_temperature() for atm in traj]
    print(temperature)




if __name__=='__main__':

    MyFirstCalculation()




















