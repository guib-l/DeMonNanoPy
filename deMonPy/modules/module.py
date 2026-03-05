#!/usr/bin/env python3
import __future__

# Import standard de python3
import os,sys
import numpy as np



import deMonPy
from deMonPy.profile import Process
from deMonPy.input import write_input
from deMonPy.output import read_output




class modules:

    def __init__(
            self,
            context=None,
            **parameters):
        
        self.context = context or None

        self.context.reset()
        self.context.update(**parameters)


    def __call__(self, **kwds):
        
        if hasattr(self, "forward"):
            self.forward(**kwds)






