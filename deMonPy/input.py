#!/usr/bin/env python3
import sys
# Import standard de python3
import os
import numpy as np


import deMonPy


from deMonPy.profile import assert_flags,exclude_flags


_atoms_symbols_to_numbers = {
    "H": 1,    "He": 2,    "Li": 3,    "Be": 4,
    "B": 5,    "C": 6,    "N": 7,    "O": 8,
    "F": 9,   "Ne": 10,    "Na": 11,    "Mg": 12,
    "Al": 13,    "Si": 14,    "P": 15,    "S": 16,
    "Cl": 17,    "Ar": 18,    "K": 19,    "Ca": 20,
    "Sc": 21,    "Ti": 22,   "V": 23,    "Cr": 24,
    "Mn": 25,    "Fe": 26,    "Co": 27,    "Ni": 28,
    "Cu": 29,    "Zn": 30,    "Ga": 31,    "Ge": 32,   
    "As": 33,    "Se": 34,    "Br": 35,    "I": 53,
}


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



    def handler_writen(self, params, bind_str=" "):
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
            elif isinstance(item,str):
                _inline.append( f"{key}{bind_str}{item}" )
            else:
                pass
        return _inline

    def _write_table(self, table, symbols=None, fmt = '10.8f'):
        """Serialize a generic table of parameters.

        Args:
            table: Mapping of table names to parameter lists.

        Returns:
            list[str]: Serialized table lines.
        """
        out = ""
        if symbols is None:
            symbols = [""] * len(table)
        for symb,values in zip(symbols, table):
            out += f"\n{symb}"
            for v in values:
                out += f" {v:{fmt}}"
        return out



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
        
        if params.pop("TRAJECTORY",False):
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

        if params.pop("TRAJECTORY",False):
            self.flags.add("traj")
            
        if "SEED" in params["MC"].keys():
            if params["MC"]["SEED"] is True:
                params["MC"]["SEED"] = np.random.randint(1,99999)

        self.io_lines["MONTECARLO"] = self.handler_writen(params.pop('MC'))

        self.io_lines["MCTEMP"] = self.handler_writen(params.pop('MCTEMP'),bind_str="=")



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
    
    
    def _io_write_mddynamics(self, params=None):
        """ Write the MDYNAMICS block, which may contain a read section for velocities.
        Args:
            params: MDYNAMICS parameter block.
        """        
        read = params.pop("READ",None)
        if "RANDOM" and "ZERO" in params.keys():
            params.pop("ZERO")
        
        if read is not None:

            veloc = False
            if hasattr(read,"keys") and "VELOCITIES" in read.keys():
                
                if read["VELOCITIES"] is not None and len(read["VELOCITIES"])>0:
                    
                    value = range(1,len(read["VELOCITIES"])+1)
                    txt = self._write_table(read["VELOCITIES"],value)
                    veloc = txt

            params["READ"] = veloc

        self.io_lines["MDYNAMICS"] = self.handler_writen(params)
        
        
    def _io_write_bath(self, params=None,):

        temp = ["SCAL","BERE","NOSE","LANGE","STOCH_R","ANDERSEN","LOCA"]

        _is_thermo = 0
        for elm in temp:
            if params[elm]: _is_thermo += 1

        if not params["NOSE"]:
            params["NTHER"] = None
            params["FREQTH"] = None

        if _is_thermo==0:
            return
        elif _is_thermo==1:
            self.io_lines["MDBATH"] = self.handler_writen(params)


    @assert_flags("md")
    def _write_md(self, params=None):
        """Write molecular dynamics directives.

        Args:
            params: Molecular dynamics parameter block.
        """        
        if params is None:
            params = self.module["MD"]
                    
        self._io_write_mddynamics(params.pop('MDYNAMICS'))

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
            self._io_write_bath(params.pop('MDBATH'))

        if 'PARATEMP' in params.keys():
            self.flags.add("ptmd")
            self.io_lines["PARATEMP"] = self.handler_writen(params.pop('PARATEMP'))

        if 'CARPAR' in params.keys():
            self.io_lines["CARPAR"] = self.handler_writen(params.pop('CARPAR'))

        if "TRAJECTORY" in params:
            if params.pop("TRAJECTORY",None):
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

        if "wmull" in self.flags:
            params.update({"WMULL":True})
        if "cm3" in self.flags:
            params.update({"CM3POT":True})
        if "cm3inter" in self.flags:
            params.update({"CM3INTER":True})
        
        self.io_lines["DFTB"] = self.handler_writen(params, bind_str="=")
        
    
    def _write_basis(self):
        """Write basis and parameter file references."""
        params = self.io_lines.pop("PARAM")
        new = [ "PTYPE="+params["PTYPE"]+f"\n{params['SKFILE']}" ]
        
        self.io_lines["PARAM"] = new
        
    @exclude_flags("molecules")
    def _write_geometry(self, symbols, positions, fmt = '%10.7f'):
        """Write the geometry block.

        Args:
            symbols: Atomic symbols.
            positions: Atomic coordinates.
            fmt: Numeric format used for coordinates.
        """
        units = ""
        if "geometry" in self.flags:
            for key,item in self.parameters["GEOMETRY"].items():
                if item:
                    units = key

        geometry = f"GEOMETRY {units}\n"

        if self.complement is None:
            self.complement = ["",]*len(symbols)

        if "QMMM" in self.flags:
            raise NotImplementedError("Flags QMMM set True")
        
        for s,p,u in zip(symbols,positions,self.complement):
            geometry += "%s %s %s %s %s\n" % \
                        (s,fmt % p[0],fmt % p[1],fmt % p[2], u)
            
        self.io_lines["GEOMETRY"] = [geometry]

        self.complement = None

    @assert_flags("molecules")
    def _write_molecules(self, fmt = '%10.7f'):
        """Write MOLECULES key-word into input file
        """

        quat = self.parameters["QUATERNION"]
        molc = self.parameters["MOLECULES"]

        geometry = ""
        coords = quat.pop("COORDS",None)

        geometry += f"QUATERNION NMOL={len(coords)}"
        for key,value in quat.items():
            if not isinstance(value,bool):
                continue
            if value:
                geometry += f' {key}'

        geometry += "\n"
        for i,_q in enumerate(coords):
            geometry += "%s %s %s %s %s %s %s %s\n" % \
                (i+1,_q[0],_q[1],_q[2],_q[3],fmt % _q[4],fmt % _q[5],fmt % _q[6],)

        geometry += f"MOLECULES NMOL={len(molc['NAMES'])}\n"
        for i,_n in enumerate(molc["NAMES"]):
            geometry += f"{i+1} {_n}\n"


        self.io_lines['GEOMETRY']  = [geometry]
        


    def _write_bondparams(self, symbols, params):

        self.io_lines["BONDPARAMS"] = []
        for key,item in params["BONDPARAMS"].items():
            elmts = key.split()
            
            if np.all([True if np.all(elm in symbols) else False for elm in elmts ]):
                self.io_lines["BONDPARAMS"].append(f"\n{str(key)} {float(item)}")

    @assert_flags("wmull")
    def _write_bondparam_wmull(self, symbols, params=None):
        """Write bond parameters matching the current element set.

        Args:
            symbols: Atomic symbols present in the system.
            params: Bond parameter block.
        """
        if params is None:
            params = self.parameters["WMULL"]
        
        self._write_bondparams(symbols, params)
        
        
    @assert_flags("cm3")
    def _write_bondparam_cm3(self, symbols, params=None):
        """Write bond parameters matching the current element set.

        Args:
            symbols: Atomic symbols present in the system.
            params: Bond parameter block.
        """
        if params is None:
            params = self.parameters.get("CM3", None )
        
        self._write_bondparams(symbols, params)
        
    @assert_flags("cm3inter")
    def _write_bondparam_cm3inter(self, symbols, params=None):
        """Write bond parameters matching the current element set.

        Args:
            symbols: Atomic symbols present in the system.
            params: Bond parameter block.
        """
        if params is None:
            params = self.parameters.get("CM3INTER", None) 
            
        self._write_bondparams(symbols, params)
    
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
            self.io_lines["FREQUENCY"] = []
        if isinstance(params, dict):
            self.io_lines[f"FREQUENCY VIB={params['VIB']}"] = []
    

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












