#!/usr/bin/env python3
import __future__

# Import standard de python3
import os
import numpy as np


import deMonPy


from deMonPy.profile import assert_flags




class write_input:

    def __init__(
            self,
            TITLE="",
            BASIS=None,
            **parameters):

        params = parameters.get("DEMON_PARAMETERS",None)
        self.parameters = params.get("ACTIVE",{})

        assert self.parameters!={}, ValueError("No parameters")

        self.io_lines = {
            "TITLE":TITLE,
            "DFTB":[],
            "PARAM":BASIS,
            "GEOMETRY":[],
        }

        self.flags = set(key.lower() for key in self.parameters.keys())
        


    def _write_basis(self):
        params = self.io_lines.pop("PARAM")
        new = [ "PTYPE="+params["PTYPE"]+f"\n{params["SKFILE"]}" ]
        
        self.io_lines["PARAM"] = new
        
    def _write_geometry(self, symbols, positions, fmt = '%10.7f'):
        
        geometry = "GEOMETRY\n"
        updt = ["",]*len(symbols)

        if "QMMM" in self.flags:
            raise NotImplementedError("Flags QMMM set True")
        
        for s,p,u in zip(symbols,positions,updt):
            geometry += "%s %s %s %s %s\n" % \
                        (s,fmt % p[0],fmt % p[1],fmt % p[2], u)
            
        self.io_lines["GEOMETRY"] = [geometry]





    @assert_flags("dftb")
    def _write_dftb(self, params=None):
        if params is None:
            params = self.parameters["DFTB"]
        
        for key,item in params.items():
            
            if item is True:
                self.io_lines['DFTB'].append(f"{key}")
            elif item > 0.0:
                self.io_lines['DFTB'].append(f"{key}={item}")
    
    @assert_flags("wmull")
    def _write_bondparam(self, params=None):
        ...
    
    @assert_flags("charge")
    def _write_charge(self, params=None):
        if params is None:
            params = self.parameters["CHARGE"]
        
        self.io_lines['CHARGE'] = {params:""}        
    
    @assert_flags("ci")
    def _write_ci(self, params=None):
        if params is None:
            params = self.parameters["CI"]

        if "CONST" in params.keys(): 
            if params["CONST"] is None:
                params.pop("CONST")
        if "CONST" not in params.keys():
            self.io_lines['DFTB'].append('CI')

        for key,item in params.items():
            
            if item is True:
                self.io_lines['DFTB'].append(f"{key}")
            elif item > 0.0:
                self.io_lines['DFTB'].append(f"{key}={item}")
        
    @assert_flags("cutsys")
    def _write_cutsys(self, params=None):
        if params is None:
            params = self.parameters["CUTSYS"]

        frags = params.pop("FRAGMENT")
        self.io_lines['CUTSYS'] = []
        self.io_lines['CUTSYS'].append(f"NMOL={len(frags)}")

        txt = ""
        for frgs in frags:
            txt += f"\n{frgs}"

        for key,item in params.items():

            if item is True:
                self.io_lines['CUTSYS'].append(f"{key}")

        self.io_lines['CUTSYS'].append(txt)
        



    
    @assert_flags("tddftb")
    def _write_tddftb(self, params=None):
        if params is None:
            params = self.parameters["TDDFTB"]

        value = params.pop("TDDFTB")
        if isinstance(value, int):
            self.io_lines['DFTB'].append(f"LRESP={value}")
        if isinstance(value, bool):
            self.io_lines['DFTB'].append("LRESP")


    
    @assert_flags("freq")
    def _write_freq(self, params=None):
        if params is None:
            params = self.parameters["FREQ"]
        
        value = params.pop("FREQ")
        if isinstance(value, bool):
            self.io_lines['DFTB'].append("FREQ")
    

    @assert_flags("qmmm")
    def _write_qmmm(self,):
        ...
    
    @assert_flags("debug")
    def _write_debug(self, params=None):
        if params is None:
            params = self.parameters["DEBUG"]



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












