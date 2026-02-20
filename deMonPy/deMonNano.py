#!/usr/bin/env python3
import __future__

# Import standard de python3
import os
import numpy as np



import deMonPy
from deMonPy.profile import Process
from deMonPy.input import write_input
from deMonPy.output import read_input



class BasicCalculation:
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



class deMonMixin:

    execut  = ""
    workdir = None

    def set_workdir(self,):

        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)

    def set_state(self, index=1, **props):
        self.state.update({"state-%s" % index: props})

    def get_state(self, index=1):
        return self.state["state-%s" % index]
    
    def to_dict(self,):
        return locals()
    
    def __repr__(self):
        import time
        txt  = "* ============================================ *\n"
        txt += f" - DEMONANO CODE at {time.asctime()}\n"
        txt += f"   > BASIS   : {self.basis} \n"
        txt += f"   > WORKDIR : {self.workdir} \n"
        txt += f"   > EXEC    : {self.execut} \n"
        txt += "* ============================================ *"
        return txt






class deMonNano(BasicCalculation,deMonMixin):

    available_properties = ["energies","forces"]


    def __init__(
            self,
            execut=None,
            workdir=".",
            omp_threads=1,
            system=True,
            prefix="DEMON",
            title="CALCULATION DEMONANO",
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

        for props in self.available_properties:
            self.results.update({ props:None })

        # Build parameters
        self.basis = parameters.pop("BASIS",basis)
        self.build(BASIS=self.basis,**parameters)




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
        
        self.set_state(
            index=index,
            **{
                "symbols":symbols,
                "positions":positions,
                "results":self.results.copy(),
                "calculator":self.to_dict()
            }
        )

    def build(self, **parameters):
        
        self.wi = write_input(**parameters)




    def write_input(
            self,
            symbols, 
            geometry):
        
        self.wi._write_dftb()
        self.wi._write_basis()
        self.wi._write_geometry(symbols,geometry)
        
        self.wi.write(
            workdir=self.workdir
        )



    def read_input(self,):
        raise NotImplemented

















