#!/usr/bin/env python3
import __future__

# Import standard de python3
import os,sys
import numpy as np



import deMonPy
from deMonPy.profile import Process
from deMonPy.input import write_input
from deMonPy.output import read_output




# Provisoir !!!
import json
import ase 

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, ase.Atoms):
            return obj.__repr__()
        return super().default(obj)

# !!!



class BasicCalculation:
    
    execut  = ""
    workdir = None

    def __init__(
            self,
            exec,
            workdir,
            prefix,
            omp_threads=1,
            system=False):
        
        self.process = Process(
            executable=exec,
            workdir=workdir,
            prefix=prefix,
            omp_threads=omp_threads,
            system=system
        )

    def execute(
            self,
            ignore_fails=False):

        try:
            self.process.execute()
        except Exception as e:
            if not ignore_fails:
                raise Exception(e)


    def set_workdir(self,):

        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)

    def set_state(self, index=1, **props):
        self.state.update({"state-%s" % index: props})

    def get_state(self, index=1):
        return self.state["state-%s" % index]
    
    def to_dict(self,):
        return locals()
    




class deMonNano(BasicCalculation):

    available_properties = ["energies","forces"]

    def __init__(
            self,
            execut=None,
            workdir=".",
            omp_threads=1,
            system=True,
            prefix="DEMON",
            title="CALCULATION DEMONANO",
            properies=['energy'],
            basis={},
            **parameters):
        
        self.parameters = parameters.copy()
        
        _execut  = parameters.pop("DEMON_EXECUTABLE",execut)
        _prefix  = parameters.pop("PREFIX",prefix)
        _workdir = parameters.pop("DEMON_WORKDIR",workdir)

        BasicCalculation.__init__(self,
                                  _execut,
                                  _workdir,
                                  _prefix,
                                  omp_threads=omp_threads,
                                  system=system )
        
        # Start running directory
        self.title   = title
        self.workdir = parameters.pop("WORKDIR",workdir)


        self.set_workdir()

        # Initialize 
        self.state   = {}
        self.results = {}

        self.flags = set()

        for props in self.available_properties:
            self.results.update({ props:None })

        # Build parameters
        self.basis = parameters.pop("BASIS",basis)

        self._wi = write_input(BASIS=self.basis,
                               **parameters)
        self.flags = self._wi.flags
        
        self._wo = read_output(properties=properies,
                               workdir=self.workdir,
                               flags=self.flags,
                               output="deMon.out")
        
    def reset(self,):
        
        self.state   = {}
        self.results = {}

        self.flags = set()

        for props in self.available_properties:
            self.results.update({ props:None })

    def update(
            self, 
            properies=['energy'],
            basis={},
            **parameters):

        # Build parameters
        self.basis = parameters.pop("BASIS",basis)

        self._wi = write_input(BASIS=self.basis,
                               **parameters)
        self.flags = self._wi.flags
        
        self._wo = read_output(properties=properies,
                               workdir=self.workdir,
                               flags=self.flags,
                               output="deMon.out")
        

    def calculate(
            self,
            *,
            symbols,
            positions,
            index=0,
            **kwargs):
        
        self.write_input(
            symbols,
            positions,)
        
        self.execute(ignore_fails=False)


        self.read_output()

        
        self.set_state(
            index=index,
            **{
                "symbols":symbols,
                "positions":positions,
                "results":self.results.copy(),
                "calculator":self.to_dict()
            }
        )

    def write_input(
            self,
            symbols, 
            geometry):
        
        # Parameters
        self._wi._write_dftb()
        self._wi._write_charge()
        self._wi._write_bondparam(symbols)
        self._wi._write_ci()
        self._wi._write_multi()
        self._wi._write_basis()
        self._wi._write_debug()
        self._wi._write_freq()
        self._wi._write_tddftb()
        self._wi._write_qmmm()
        self._wi._write_cutsys()

        # Modules
        self._wi._write_opt()
        self._wi._write_ptmc()
        self._wi._write_md()
        self._wi._write_neb()
        
        # Geometry writting
        self._wi._write_geometry(symbols=symbols,
                                positions=geometry)
        
        self._wi.write(
            workdir=self.workdir
        )


    def read_output(self,):

        # Parameters
        self._wo.read_file()
        self._wo.read_freq()
        
        self._wo.read_energy()
        self._wo.read_ci()
        self._wo.read_tddftb()

        self._wo.read_debug()

        # Modules
        self._wo._read_opt()
        self._wo._read_ptmc()
        self._wo._read_md()
        self._wo._read_neb()

        # Geometry reading
        self._wo.read_geometry(output='deMon.mol',
                               is_charges=False, 
                               keep=1,)
        
        self.results = self._wo.complet_results

    def print_results(self, files=sys.stdout):
        print(
            json.dumps(
                self._wo.complet_results, 
                indent=4, 
                ensure_ascii=True, 
                cls=NumpyEncoder
            ),
            file=files
        )



from deMonPy import available_modules


class Module_DeMonNano(deMonNano):


    def __init__(
            self,
            module=None,
            execut=None,
            workdir=".",
            omp_threads=1,
            system=True,
            prefix="DEMON",
            title="CALCULATION MODULE-DEMONANO",
            properies=['energy'],
            basis={},
            available_modules=available_modules,
            **parameters):  
        
        super().__init__(
            execut=execut,
            workdir=workdir,
            omp_threads=omp_threads,
            system=system,
            prefix=prefix,
            title=title,
            properies=properies,
            basis=basis,
            **parameters
        )

        self.module   = module

        self.is_build = False
        self.build    = None


    @property
    def module(self):
        return self._module

    @module.setter
    def module(self, module):
        if module not in available_modules.keys():
            raise NotImplementedError(f"{module} is not available")
        self._module = available_modules[module]


    def initialize(self, **kwds):
        params = self.parameters
        module = self._module.pop("module", None)
        args   = self._module.pop("args", {})

        assert module is not None, ValueError("Unknow module")
        
        args.update(**kwds)
        params.update(**args)

        print(
            json.dumps( 
                params,
                indent=4, 
            )
        )

        self.build = module(context=self, **params)
        self.is_build = True


    def reset(self):
        self.is_build = False
        self.build    = None


    def __call__(self, **kwds):
        
        if not self.is_build:
            self.initialize()

        return self.build.__call__(**kwds)





