#!/usr/bin/env python3
import __future__

# Import standard de python3
import os,sys
import numpy as np



import deMonPy
from deMonPy.profile import Process
from deMonPy.input import write_input
from deMonPy.output import read_output


from deMonPy.modules.module import modules



class _relax_geometry(modules):

    def __init__(
            self,
            context,
            **kwargs):
        
        super().__init__(context=context, **kwargs)

        self._module_parameters = None


    def restart(self, **kwds):

        image = kwds.pop("image", None)
        if not image:
            image = self.context.results["output_geometry"]
        
        self._module_parameters.update(**kwds)
        
        self.forward(
            image=image,
            **self._module_parameters
        )

    def check_distances(self,):
        pass

    def is_converged(self,):

        for line in self.context._wo.lines:
            if self.context._wo.is_inside(
                    "optimization not converged",line ):
                return False
            
        return True


    def update_parameters(self, kwds):

        params = self.context.parameters
        params.update(kwds)
        self.context.update(**params)

    def forward(
            self, 
            image,
            max=999,
            algo='CGRAD',
            out=1,
            restart=False,
            **args):
        
        self._module_parameters = dict(
            max=999,
            algo='CGRAD',
            out=1,
            restart=False,
            **args
        )

        self.update_parameters({
                "DEMON_MODULE":{
                    "ACTIVE":{
                        'OPT':{
                            "MAX":max,
                            algo:True,
                            "OUT":out,
                            **args
                        } } } }
        )

        self.context.calculate(
            symbols=image.symbols,
            positions=image.positions
        )
        
        if not self.is_converged():
            self.context.results["converged"]=False
        else:
            self.context.results["converged"]=True










