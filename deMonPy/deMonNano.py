#!/usr/bin/env python3
import __future__

# Import standard de python3
import os
import numpy as np




import deMonPy
from deMonPy.profile import process
from deMonPy.input import write_input
from deMonPy.output import read_input




class deMonMixin:

    execut  = ""
    workdir = None

    def set_workdir(self,):

        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)



class deMonNano(deMonMixin):

    available_properties = ["energies","forces"]

    _results = {}

    def __init__(
            self,
            title="CALCULATION DEMONANO",
            basis={},
            execut=None,
            workdir=".",
            **parameters):
        
        self.workdir = workdir
        self.execut = execut

        self.set_workdir()

        for props in self.available_properties:
            self._results.update({ props:None })


    def calculate(
            self,
            image,):
        pass

    def write_input(self,):
        raise NotImplemented
    
    def read_input(self,):
        raise NotImplemented

















