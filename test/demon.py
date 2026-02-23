#!/usr/bin/env python3
import __future__

import os
import sys

import configs

import deMonPy
from deMonPy.profile import read_json
from deMonPy.deMonNano import deMonNano


import numpy as np
import ase 
from ase.atoms import Atoms

import json

parameter_config = read_json("./config.json")


if __name__=='__main__':

    image = Atoms(
        ["O","H","H","O","H","H"],
        positions=np.array(
            [[1.2478,-0.5185,3.4049],
             [1.5946,-1.4204,3.3886],
             [0.9008,-0.3341,2.5062],
             [3.2478,-0.5185,3.4049],
             [3.5946,-1.4204,3.3886],
             [2.9008,-0.3341,2.5062],
             ])
        )

    dem = deMonNano(
        title="CALCULATION DEMONANO",
        basis={},
        execut="~/Documents/dev_deMon/deMon.x",
        workdir=".run/",
        **parameter_config
    )

    dem.calculate(
        symbols=image.symbols,
        positions=image.positions
    )


    




    sys.exit()





