#!/usr/bin/env python3
import __future__

# Import standard de python3
import os
import numpy as np


import deMonPy




class write_input:

    def __init__(
            self,
            TITLE="",
            BASIS=None,
            **parameters):

        params = parameters.get("DEMON_PARAMETERS",None)
        self.parameters = params.get("ACTIVE",{})

        self.flags = set()

        assert self.parameters!={}, ValueError("No parameters")

        self.io_lines = {
            "TITLE":TITLE,
            "DFTB":[],
            "PARAM":BASIS,
            "GEOMETRY":[],
        }

    def _write_dftb(self, params=None):
        if params is None:
            params = self.parameters["DFTB"]
        
        for key,item in params.items():
            
            if item is True:
                self.io_lines['DFTB'].append(f"{key}")
            elif item > 0.0:
                self.io_lines['DFTB'].append(f"{key}={item}")


    def _write_basis(self):
        params = self.io_lines.pop("PARAM")
        new = [
            "PTYPE="+params["PTYPE"]+f"\n{params["SKFILE"]}"
        ]
        self.io_lines["PARAM"] = new
        
        


    def _write_geometry(self, symbols, positions, fmt = '%10.7f'):
        
        geometry = "GEOMETRY\n"
        updt = ["",]*len(symbols)

        if "qmmm" in self.flags:
            raise NotImplementedError("Flags QMMM set True")
        
        for s,p,u in zip(symbols,positions,updt):

            geometry += "%s %s %s %s %s\n" % \
                        (s,fmt % p[0],fmt % p[1],fmt % p[2], u)
            
        self.io_lines["GEOMETRY"] = [geometry]

    
    def _write_bondparam(self,):
        ...
    
    def _write_ci(self,):
        ...
    
    def _write_tddftb(self,):
        ...
    
    def _write_cutsys(self,):
        ...
    
    def _write_freq(self,):
        ...
    
    def _write_qmmm(self,):
        self.flags.add("qmmm")
        ...
    
    def _write_debug(self,):
        ...

    def write(self, 
              input="deMon.inp",
              workdir=""):
        
        path = os.path.join(workdir,input)

        with open(path, "w") as fd:
            
            geom = self.io_lines.pop("GEOMETRY")
            for key,item in self.io_lines.items():
                
                fd.write(key)
                
                for elm in item:
                    fd.write(f" {elm}")
                fd.write("\n")

            fd.write(geom[-1])












