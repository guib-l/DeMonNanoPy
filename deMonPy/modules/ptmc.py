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


class _ptmc(modules):

    def __init__(
            self,
            context,
            **kwargs):
        
        super().__init__(context=context, **kwargs)

        self._module_parameters = None

    def ptmc(self,
             **kwds):
        
        return self.forward(**kwds)

    def mc(self,
           **kwds):
        kwds.pop("max_temp",None)
        kwds.pop("min_temp",None)
        return self.forward(**kwds)

    def forward(
            self, 
            image,
            max=30,
            seed=True,
            out=1,
            wall=8.0,
            temperature=None,
            n_temp=10,
            max_temp=300,
            min_temp=30,
            rescale=10,
            distribution_temp="geom",
            temp_list=None,
            restart=False,
            **args):
        
        self._module_parameters = dict(
            max=max,
            seed=seed,
            out=out,
            restart=restart,
            temperature=temperature,
            n_temp=n_temp,
            max_temp=max_temp,
            min_temp=min_temp,
            distribution_temp=distribution_temp,
            rescale=rescale,
            temp_list=temp_list,
            **args
        )

        assert temp_list is None, NotImplementedError("LIST is not implemented")

        assert distribution_temp.lower() in ["geom","linear"], \
            ValueError("Unknow temperature distribution : (any of 'geom' or 'linear')")

        self.update_parameters({
                "DEMON_MODULE":{
                    "ACTIVE":{
                        'PTMC':{
                            "MC":{
                                "MAX":max,
                                "SEED":seed,
                                "WALL":wall
                            },
                        "MCTEMP":{
                            "TMC":None,
                            "NTEMP":n_temp,
                            distribution_temp.upper():True,
                            "TEMPMIN":min_temp,
                            "TEMPMAX":max_temp,
                            "RESCALE":rescale,
                            "SDBG":False,
                            "OUT":1,
                            "SOUT":1
                        }
                    } } } }
        )

        self.context.calculate(
            symbols=image.symbols,
            positions=image.positions
        )
        










