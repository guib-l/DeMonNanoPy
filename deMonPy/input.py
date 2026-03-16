#!/usr/bin/env python3
import __future__

import sys
# Import standard de python3
import os
import numpy as np


import deMonPy


from deMonPy.profile import assert_flags

def parse_range_string(range_string: str) -> list[int]:
    """Expand a comma-separated range expression into a list of integers.

    Args:
        range_string: String containing integers or inclusive ranges such as
            ``"1,3-5,8"``.

    Returns:
        list[int]: Expanded integer values.
    """
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
    """Build deMonNano input sections from user parameters."""

    def __init__(
            self,
            TITLE="",
            BASIS=None,
            **parameters):
        """Initialize the input writer.

        Args:
            TITLE: Title written to the input file.
            BASIS: Basis and parameter definition.
            **parameters: Full deMonNano configuration dictionary.
        """

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

        try:
            self.flags.add(*[key.lower() for key in self.module.keys()])
        except:
            pass

        self.complement = None



    def handler_writen(self, params):
        """Convert a parameter dictionary into inline deMonNano tokens.

        Args:
            params: Mapping of parameter names to values.

        Returns:
            list[str]: Tokens ready to be written in the input file.
        """
        
        _inline = []

        for key,item in params.items():

            if isinstance(item,dict):
                _inline.append( self.handler_writen(item) )
            elif item is None:
                continue
            elif isinstance(item,bool):
                if item:
                    _inline.append( str(key) )
            elif isinstance(item,(float,int)):
                _inline.append( f"{key}={item}" )
            else:
                pass
        return _inline




    # =========================================================================
    # =========== MODULES -----------------------------------------------------
    # =========================================================================



    @assert_flags("opt")
    def _write_opt(self, params=None):
        """Write optimization-related input directives.

        Args:
            params: Optimization parameter block.
        """
        if params is None:
            params = self.module["OPT"]

        self.flags.remove("opt")
        if "TRAJECTORY" in params:
            if params.pop("TRAJECTORY",None):
                self.flags.add("traj")
                
        self.io_lines["OPTIMIZATION"] = []

        for key,item in params.items():
            
            if item is True:
                self.io_lines['OPTIMIZATION'].append(f"{key}")
            elif item > 0.0:
                self.io_lines['OPTIMIZATION'].append(f"{key}={item}")
        


    @assert_flags("ptmc")
    def _write_ptmc(self, params=None):
        """Write PTMC module directives.

        Args:
            params: PTMC parameter block.
        """
        if params is None:
            params = self.module["PTMC"]

        if "SEED" in params["MC"].keys():
            if params["MC"]["SEED"] is True:
                params["MC"]["SEED"] = np.random.randint(1,99999)

        self.io_lines["MONTECARLO"] = self.handler_writen(params.pop('MC'))

        self.io_lines["MCTEMP"] = self.handler_writen(params.pop('MCTEMP'))



    def _write_constraint(self, constraint):
        """Serialize molecular dynamics constraints.

        Args:
            constraint: Constraint mapping indexed by direction.

        Returns:
            list[str]: Serialized constraint lines.
        """
        out = ""
        for dir,value in constraint.items():
            value = parse_range_string(value)
            
            out = f"\n{dir} {value[0]}"
            for v in value[1:]:
                out += f" {v}"
        return [out]

    @assert_flags("md")
    def _write_md(self, params=None):
        """Write molecular dynamics directives.

        Args:
            params: Molecular dynamics parameter block.
        """
        if params is None:
            params = self.module["MD"]
        
        self.io_lines["MDYNAMICS"] = self.handler_writen(params.pop('MDYNAMICS'))

        self.io_lines["TIMESTEP"] = [str(params.pop('TIMESTEP'))]

        mdtemp = params.pop('MDTEMP',None)
        if mdtemp is not None:
            self.io_lines["MDTEMP"] = [str(mdtemp)]

        self.io_lines["MDSTEP"] = self.handler_writen(params.pop('MDSTEP'))

        if 'MDCONSTRAINTS' in params.keys():
            self.io_lines["MDCONSTRAINTS"] =self._write_constraint(params.pop('MDCONSTRAINTS'))

        if 'CONSERVE' in params.keys():
            self.io_lines["CONSERVE"] = self.handler_writen(params.pop('CONSERVE'))

        if 'MDBATH' in params.keys():
            self.io_lines["MDBATH"] = self.handler_writen(params.pop('MDBATH'))

        if 'CARPAR' in params.keys():
            self.io_lines["CARPAR"] = self.handler_writen(params.pop('CARPAR'))

        if "TRAJECTORY" in params:
            if params["TRAJECTORY"]:
                self.flags.add("traj")


    @assert_flags("neb")
    def _write_neb(self, params=None):
        """Write nudged elastic band directives.

        Args:
            params: NEB parameter block.
        """
        if params is None:
            params = self.module["NEB"]




    # =========================================================================
    # =========== PARAMETERS --------------------------------------------------
    # =========================================================================

    @assert_flags("dftb")
    def _write_dftb(self, params=None):
        """Write the main DFTB parameter section.

        Args:
            params: DFTB parameter block.
        """
        if params is None:
            params = self.parameters["DFTB"]
        
        self.io_lines["DFTB"] = self.handler_writen(params)
        
    
    def _write_basis(self):
        """Write basis and parameter file references."""
        params = self.io_lines.pop("PARAM")
        new = [ "PTYPE="+params["PTYPE"]+f"\n{params["SKFILE"]}" ]
        
        self.io_lines["PARAM"] = new
        
    def _write_geometry(self, symbols, positions, fmt = '%10.7f'):
        """Write the geometry block.

        Args:
            symbols: Atomic symbols.
            positions: Atomic coordinates.
            fmt: Numeric format used for coordinates.
        """

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
        """Write bond parameters matching the current element set.

        Args:
            symbols: Atomic symbols present in the system.
            params: Bond parameter block.
        """
        if params is None:
            params = self.parameters["WMULL"]
        
        self.io_lines["BONDPARAMS"] = []
        for key,item in params["BONDPARAMS"].items():
            elmts = key.split()
            
            if np.all([True if np.all(elm in symbols) else False for elm in elmts ]):
                self.io_lines["BONDPARAMS"].append(f"\n{str(key)} {float(item)}")
        
        
    
    @assert_flags("charge")
    def _write_charge(self, params=None):
        """Write the total charge section.

        Args:
            params: Charge value or charge descriptor.
        """
        if params is None:
            params = self.parameters["CHARGE"]
        
        self.io_lines['CHARGE'] = {params:""}        
    
    
    @assert_flags("multi")
    def _write_multi(self, params=None):
        """Write the multiplicity section.

        Args:
            params: Spin multiplicity value or descriptor.
        """
        if params is None:
            params = self.parameters["MULTI"]
        
        self.io_lines['MULTI'] = {params:""}        

    @assert_flags("ci")
    def _write_ci(self, params=None):
        """Write configuration interaction directives.

        Args:
            params: Configuration interaction parameter block.
        """
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
        """Write subsystem fragmentation directives.

        Args:
            params: CUTSYS parameter block.
        """
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
        """Write TD-DFTB response options.

        Args:
            params: TD-DFTB parameter value or block.
        """
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
        """Write frequency analysis directives.

        Args:
            params: Frequency analysis configuration.
        """
        if params is None:
            params = self.parameters["FREQ"]
        
        if isinstance(params, bool):
            self.io_lines['FREQUENCY'] = []
        if isinstance(params, float):
            self.io_lines[f'FREQUENCY={params}'] = []
    

    @assert_flags("qmmm")
    def _write_qmmm(self, params=None):
        """Write QM/MM configuration and atom partitioning.

        Args:
            params: QM/MM parameter block.
        """
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




    @assert_flags("print")
    def _write_debug(self, params=None):
        """Write debug and print directives.

        Args:
            params: Print/debug parameter block.
        """
        if params is None:
            params = self.parameters["PRINT"]

        self.io_lines["PRINT"] = self.handler_writen(params)



    # =========================================================================
    # =========== WRITABLE ----------------------------------------------------
    # =========================================================================

    def write(self, 
              input="deMon.inp",
              workdir=""):
        """Write the assembled input file to disk.

        Args:
            input: Output input file name.
            workdir: Directory where the file is written.
        """
        
        path = os.path.join(workdir,input)

        with open(path, "w") as fd:
            
            geom = self.io_lines.pop("GEOMETRY")
            for key,item in self.io_lines.items():
                
                fd.write(key)
                
                for elm in item:
                    fd.write(f" {elm}")
                fd.write("\n")

            fd.write(geom[-1])












