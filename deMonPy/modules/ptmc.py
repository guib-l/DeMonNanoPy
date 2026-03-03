#!/usr/bin/env python3
import __future__

# Import standard de python3
import os,sys
import numpy as np



import deMonPy
from deMonPy.profile import Process
from deMonPy.input import write_input
from deMonPy.output import read_output


from deMonPy.modules.module import Modules


class _ptmc(Modules):

    def __init__(
            self,
            **kwargs):
        

        super().__init__(self, **kwargs)


    def forward(self,):

        pass










