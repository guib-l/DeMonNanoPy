#!/usr/bin/env python3
import __future__

# Import standard de python3
import os,sys


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






