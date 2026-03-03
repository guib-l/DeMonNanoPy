#!/usr/bin/env python3
import __future__

# Import standard de python3
import os,sys
import numpy as np



import deMonPy
from deMonPy.profile import Process
from deMonPy.input import write_input
from deMonPy.output import read_output



class Modules:

    def __init__(
            self,
            deMonNano_object,
            **parameters):
        
        self._demon = deMonNano_object
        self._demon.update(**parameters)

    

    def forward(self, **kwargs):
        ...



    def __call__(self, *args, **kwds):
        
        self.forward(**kwds)




