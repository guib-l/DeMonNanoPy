#!/usr/bin/env python3
import __future__

# Import standard de python3
import os,sys
import copy
from copy import deepcopy


class modules:

    def __init__(
            self,
            context=None,
            **parameters):
        
        self.context = context or None

        self.context.reset()
        self.context.update(**parameters)

    def update_parameters(self, kwds):

        def recursive_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = recursive_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        params = copy.deepcopy(self.context.parameters)
        params = recursive_update(params, kwds)
        
        self.context.update(**params)

    def __call__(self, **kwds):
        
        if hasattr(self, "forward"):
            self.forward(**kwds)






