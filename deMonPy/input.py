#!/usr/bin/env python3
import __future__

import sys
# Import standard de python3
import os
import numpy as np


import deMonPy


from deMonPy.profile import assert_flags

def parse_range_string(range_string: str) -> list[int]:
    if not range_string:
        return []
    result = []

    for part in range_string.split(","):
        part = part.strip()

        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)

            step = 1 if start <= end else -1
            result.extend(range(start, end + step, step))
        else:
            result.append(int(part))

    return result


class write_input:

    def __init__(
            self,
            TITLE="",
            BASIS=None,
            **parameters):

        self.io_lines = {
            "TITLE":TITLE,
            "DFTB":[],
            "PARAM":BASIS,
            "GEOMETRY":[],
        }

        # Update Parameters Flags
        params = parameters.get("DEMON_PARAMETERS",None)
        self.parameters = params.get("ACTIVE",{})

        assert self.parameters!={}, ValueError("No parameters")
        self.flags = set(key.lower() for key in self.parameters.keys())
        
        # Update Modules Flags
        params = parameters.get("DEMON_MODULE",{})
        self.module = params.get("ACTIVE",{})

        self.flags.add(*[key.lower() for key in self.module.keys()])
        self.complement = None

        print(self.flags)



    # =========================================================================
    # =========== MODULES -----------------------------------------------------
    # =========================================================================

    @assert_flags("opt")
    def _write_opt(self, params=None):
        if params is None:
            params = self.module["OPT"]

        self.io_lines["OPTIMIZATION"] = []

        for key,item in params.items():
            
            if item is True:
                self.io_lines['OPTIMIZATION'].append(f"{key}")
            elif item > 0.0:
                self.io_lines['OPTIMIZATION'].append(f"{key}={item}")
        
        self.flags.remove("opt")
        if "TRAJECTORY" in params:
            if params["TRAJECTORY"]:
                self.flags.add("traj")


    @assert_flags("ptmc")
    def _write_ptmc(self, params=None):
        if params is None:
            params = self.module["PTMC"]

    @assert_flags("md")
    def _write_md(self, params=None):
        if params is None:
            params = self.module["MD"]

    @assert_flags("neb")
    def _write_neb(self, params=None):
        if params is None:
            params = self.module["NEB"]




    # =========================================================================
    # =========== PARAMETERS --------------------------------------------------
    # =========================================================================

    @assert_flags("dftb")
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
        new = [ "PTYPE="+params["PTYPE"]+f"\n{params["SKFILE"]}" ]
        
        self.io_lines["PARAM"] = new
        
    def _write_geometry(self, symbols, positions, fmt = '%10.7f'):
        
        geometry = "GEOMETRY\n"

        if self.complement is None:
            self.complement = ["",]*len(symbols)

        if "QMMM" in self.flags:
            raise NotImplementedError("Flags QMMM set True")
        
        for s,p,u in zip(symbols,positions,self.complement):
            geometry += "%s %s %s %s %s\n" % \
                        (s,fmt % p[0],fmt % p[1],fmt % p[2], u)
            
        self.io_lines["GEOMETRY"] = [geometry]

        self.complement = None

        
    @assert_flags("wmull")
    def _write_bondparam(self, symbols, params=None):
        if params is None:
            params = self.parameters["WMULL"]
        
        self.io_lines["BONDPARAM"] = []
        for key,item in params["BONDPARAMS"].items():
            elmts = key.split()
            
            if np.all([True if np.all(elm in symbols) else False for elm in elmts ]):
                self.io_lines["BONDPARAM"].append(f"\n{str(key)} {float(item)}")
        
        
    
    @assert_flags("charge")
    def _write_charge(self, params=None):
        if params is None:
            params = self.parameters["CHARGE"]
        
        self.io_lines['CHARGE'] = {params:""}        
    
    
    @assert_flags("multi")
    def _write_multi(self, params=None):
        if params is None:
            params = self.parameters["MULTI"]
        
        self.io_lines['MULTI'] = {params:""}        

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
        
    
    @assert_flags("td-dftb")
    def _write_tddftb(self, params=None):
        if params is None:
            params = self.parameters["TD-DFTB"]

        value = params
        if isinstance(value, bool):
            if value:
                self.io_lines['DFTB'].append("LRESP")
            else:
                self.flags.remove("td-dftb")
        elif isinstance(value, int):
            self.io_lines['DFTB'].append(f"LRESP={value}")

    
    @assert_flags("freq")
    def _write_freq(self, params=None):
        if params is None:
            params = self.parameters["FREQ"]
        
        value = params.pop("FREQ")
        if isinstance(value, bool):
            self.io_lines['DFTB'].append("FREQ")
    

    @assert_flags("qmmm")
    def _write_qmmm(self, params=None):
        if params is None:
            params = self.parameters["QMMM"]

        self.io_lines['QMMM'] = ["QM/MM"]

        if "RG" in params.keys():
            rg    = params["RG"].upper()
            self.io_lines['QMMM'].append(f"COUPLING={rg}")

            polaqm = params.get("polaqm".upper(),True)
            polamm = params.get("polamm".upper(),True)

            if not polaqm:
                self.io_lines['DFTB'].append("NOPOLQM")
            if not polamm:
                self.io_lines['DFTB'].append("NOPOLMM")
            
            alpha = params.get("alpha".upper(),0.0)
            self.io_lines['DFTB'].append(f"ALPHARG={alpha}")


        qm = parse_range_string(params["QM"])
        mm = parse_range_string(params["MM"])
        
        self.complement = ["",] * sum(qm+mm)
        for idx in qm:
            self.complement[idx] = "Q=0.0 QMMM=QM"
        for idx in mm:
            self.complement[idx] = "Q=0.0 QMMM=MM"




    @assert_flags("debug")
    def _write_debug(self, params=None):
        if params is None:
            params = self.parameters["DEBUG"]





    # =========================================================================
    # =========== WRITABLE ----------------------------------------------------
    # =========================================================================

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












