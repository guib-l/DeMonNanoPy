#!/usr/bin/env python3
import __future__

# Import standard de python3
import os,sys
import numpy as np



import deMonPy
from deMonPy.profile import Process
from deMonPy.input import write_input
from deMonPy.output import read_output


from deMonPy.modules import (
    ptmc, quench, ionization
)

_available_modules = {
    "ptmc":{
        "object":ptmc._ptmc,
        "default_args":{},
    },
    "rlx":{
        "object":quench._relax_geometry,
        "default_args":{},
    },
    "ips":{
        "object":ionization._ionization_potential,
        "default_args":{},
    },
    "neb":{
        "object":None,
        "default_args":{},
    },
    "dyn":{
        "object":None,
        "default_args":{},
    },
    "ptmd":{
        "object":None,
        "default_args":{},
    },
    "freq":{
        "object":None,
        "default_args":{},
    },
}