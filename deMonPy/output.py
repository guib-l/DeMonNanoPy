#!/usr/bin/env python3
import __future__

# Import standard de python3
import os
import numpy as np


import deMonPy



class IOread(object):

    def __init__(self,):

        self._block = ""
        self._tocken_block = False

        self.counter = 0

    def compute_block(self, line, 
            ctrl_in="", ctrl_out="", nb_line=None, add=1 ):

        if self.is_inside(control=ctrl_in, line=line ):
            self._tocken_block = True
            self.counter = nb_line

        if self.counter > 0:
            self._block  += line
            self.counter -= 1
        else:
            self._tocken_block = False

    def get_float(self, line, index=-1):
        assert index!=None, ValueError("index need to ba an integer")
        return float(line.split()[index])

    def get_int(self, line, index=-1):
        assert index!=None, ValueError("index need to ba an integer")
        return int(line.split()[index])

    def get_list(self, line, index_in=None, index_out=None, ctype=np.float64):
        if index_in==None:
            return ctype( line.split()[:index_out] )
        if index_out==None:
            return ctype( line.split()[index_in:] )
        return ctype( line.split()[index_in:index_out] )
        
    def get_dict(self,):
        raise NotImplementedError

    def get_string(self, line, index=None):
        if index==None:
            return line
        return str(line.split()[index])


    def is_inside(self, control="", line="" ): 
        if control in line: return  True
        else: False





class read_input:

    def __int__(
            self,):
        pass




