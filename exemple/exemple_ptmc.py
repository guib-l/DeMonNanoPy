import sys 

import pytest
import configs
import numpy as np

import copy
from copy import deepcopy

from ase.atoms import Atoms
from deMonPy.deMonNano import deMonNano
from deMonPy.deMonNano import Module_DeMonNano

parameters = {
    "DEMON_EXECUTABLE":configs.EXECUTABLE,
    "BASIS":{
        "PTYPE":"BIO",
        "SKFILE":"../../basis"
    },
    "DEMON_PARAMETERS":{
        "ACTIVE":{
            "DFTB":{
                "SCC":True,
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


WORKDIR = ".run/ptmc/"


def exemple_run_ptmc():

    mod = Module_DeMonNano(
        module="ptmc", 
        title="CALCULATION DEMONANO",
        basis={},
        execut="~/Documents/dev_deMon/deMon.x",
        workdir=WORKDIR,
        **parameters
    )

    # Run PTMC
    mod(image=image, max=30,)

    # Print results
    mod.print_results()

def exemple_run_mc():

    mod = Module_DeMonNano(
        module="ptmc", 
        title="CALCULATION DEMONANO",
        basis={},
        execut="~/Documents/dev_deMon/deMon.x",
        workdir=WORKDIR,
        **parameters
    )

    # Run PTMC
    mod(method="mc",image=image, max=30,)

    # Print results
    mod.print_results()




if __name__=='__main__':

    


    exemple_run_mc()
    exemple_run_ptmc()

    sys.exit()





