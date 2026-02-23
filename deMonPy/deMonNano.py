#!/usr/bin/env python3
import __future__

# Import standard de python3
import os
import numpy as np



import deMonPy
from deMonPy.profile import Process
from deMonPy.input import write_input
from deMonPy.output import read_output








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
    







available_flags = {
    "qmmm":"QMMM",
    "dftb":"DFTB",
    "tddftb":"TD-DFTB",
    "ci":"CI",
    "":"FREQ",
    "":"CM3",
    "":"CHARGE",
    "":"MULTI",
    "":"CUTSYS",
    "":"DEBUG",
    "":"DIPOLE",
    "":"OPT",
    "":"MD",
    "":"PTMC",
}


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
        

        BasicCalculation.__init__(self,
                                  execut,workdir,prefix,
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


    def __repr__(self):
        import time
        txt  = "* ============================================ *\n"
        txt += f" - DEMONANO CODE at {time.asctime()}\n"
        txt += f"   > BASIS   : {self.basis} \n"
        txt += f"   > WORKDIR : {self.workdir} \n"
        txt += f"   > EXEC    : {self.execut} \n"
        txt += "* ============================================ *"
        return txt





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
        
        self._wi._write_dftb()
        self._wi._write_basis()
        self._wi._write_geometry(symbols=symbols,
                                positions=geometry)
        
        
        self._wi._write_qmmm()
        
        self._wi.write(
            workdir=self.workdir
        )



    def read_output(self,):

        self._wo.read_file()
        self._wo.read_geometry(output='deMon.mol',
                               is_charges=False, 
                               keep=1,)
        self._wo.read_energy()


        print(self._wo.complet_results)













