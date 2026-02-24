#!/usr/bin/env python3
import __future__

# Import standard de python3
import os
import numpy as np


import deMonPy
from deMonPy.molden import read_XYZ
from deMonPy.profile import assert_flags



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





class read_output(IOread):

    _criteria_energy_str = {
        "DFTB total energy":"energy",
        "DFTB electronic energy": "electronic_energy",
        "DFTB band energy"      : "band_energy",
        "DFTB repulsive energy" : "repulsive_energy",
        "DFTB Coulomb energy"   : "coulomb_energy",
        "DFTB London energy"        : "london_energy",
        "DFTB MM+Mechanical Coupl." : "MM_coupl.",
        "DFTB electronic entropy"   : "electronic_entropy",
        "DFTB Polarisation energy"  : "polarisation_energy",
        "DFTB HOMO-LUMO gap"        : "HOMO-LUMO_gap",
        "DFTB (HOMO)-(HOMO-1) gap"  : "HOMO_gap",
        "DFTB Fermi energy level"   : "fermi_energy", 
        "DFTB third order Coulomb energy" : "3d_coulomb_energy",
    }

    
    def __init__(self, 
            properties=["energy"], 
            workdir="./",
            output="deMon.out", 
            flags=set(),
            type_calculation={}, ): 
        
        IOread.__init__(self, )


        self.workdir = workdir
        self.output = output

        self.flags = flags

        self.properties = properties
        self.complet_results = {}
        
        self.lines = []



    def read_file(self,):

        filename = os.path.join(self.workdir,self.output)
        with open(filename,'r') as fd:

            for line in fd.readlines():
                self.lines.append(line)




    def read_energy(self):

        start_search = False
        
        for line in self.lines:
                        
            _energy = self.get_energies(line,start_search=start_search)
            self.complet_results.update(_energy)

            if 'energy' in self.complet_results.keys():
                start_search = True
    
    def get_energies(self, 
                     line, 
                     criteria="DFTB total energy",
                     start_search=False):
        
        _energy = {}

        if self.is_inside(criteria, line):
            label = self._criteria_energy_str[criteria]
            _energy[label] = self.get_float( line, index=-1)
            
        if start_search:
            
            for cs,it in self._criteria_energy_str.items():
                if self.is_inside(cs, line):
                    _energy[it] = self.get_float(line, index=-1) 
        return _energy




    def read_ci(self):
        start_search = False

        _state = {}
        count = 0
        
        for line in self.lines:
                        
            _energy = self.get_energies(line,start_search=start_search)
            _state.update(_energy)

            if 'energy' in _state.keys():
                start_search = True
                

            if "*********   CONFIGURATIONS   *********" in line:
                count += 1
                self.complet_results[f"state_{count}"] = _state
                _state = {}
                start_search = False


        

    def read_tddftb(self):
        pass

    def read_freq(self):
        pass

    def read_debug(self):
        pass

    def read_geometry(self, output='deMon.mol',is_charges=False, keep=1):

        
        filename = os.path.join(self.workdir,output)
        data,info = read_XYZ(filename,is_charges=is_charges, keep=keep)

        if len(data)==2:
            self.complet_results["input_geometry"]  = data[0]
            self.complet_results["output_geometry"] = data[-1]

        if  len(data)>2:
            self.complet_results["input_geometry"]  = data[0]
            self.complet_results["trajectory"] = data
            self.complet_results["output_geometry"] = data[-1]











