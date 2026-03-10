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


class _dyn(modules):

    def __init__(
            self,
            context,
            **kwargs):
        
        super().__init__(context=context, **kwargs)

        self._module_parameters = None


    def forward(
            self, 
            image,
            restart=False,
            **args):
        
        self._module_parameters = dict(
            restart=restart,
            **args
        )

        self.update_parameters({
                "DEMON_MODULE":{
                    "ACTIVE":{

                    } } } 
        )


        self.context.calculate(
            symbols=image.symbols,
            positions=image.positions
        )
        
