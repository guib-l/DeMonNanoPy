#!/usr/bin/env python3
import __future__

# Import standard de python3
import os,sys
import numpy as np



import deMonPy
from deMonPy.profile import Process
from deMonPy.input import write_input
from deMonPy.output import read_output


from deMonPy.modules.module import modules


class _dyn(modules):

    def __init__(
            self,
            context,
            **kwargs):
        
        super().__init__(context=context, **kwargs)

        self._module_parameters = None

    def update_mddynamics(self, velocities=None,temp=300, wall=None):

        params = self.context.parameters["DEMON_MODULE"]["ACTIVE"]["MD"]['MDYNAMICS']

        params["READ"] = {"VELOCITIES": velocities} if velocities is not None else False
        params["RANDOM"] = temp 
        params["WALL"] = wall


    def update_mdstep(self, max=100, out=1):
        params = self.context.parameters["DEMON_MODULE"]["ACTIVE"]["MD"]['MDSTEP']
        params["OUT"] = out
        params["MAX"] = max

    def update_time_step(self, timestep=0.4):
        params = self.context.parameters["DEMON_MODULE"]["ACTIVE"]["MD"]['TIMESTEP'] = timestep

    def add_trajectory_output(self, out_traj=True):
        params = self.context.parameters["DEMON_MODULE"]["ACTIVE"]["MD"]
        params["TRAJECTORY"] = out_traj


    def restart(self, **kwds):

        image = kwds.pop("image", None)
        if not image:
            image = self.context.results["output_geometry"]

        try:
            velocities = image.get_velocities()
        except Exception as e:
            print(f"Warning: Could not retrieve velocities from the image. Error: {e}")
            velocities = None
        
        kwds["velocities"] = velocities
        
        self._module_parameters.update(**kwds)
        
        self.forward(
            image=image,
            **self._module_parameters
        )

    def forward(
            self, 
            image,
            restart=False,
            temp=300,
            velocities=None,
            timestep=0.4,
            max_steps=100,
            wall=None,
            out=10,
            out_traj=True,
            **args):
        
        
        self._module_parameters = dict(
            restart=restart,
            velocities=velocities,
            temp=temp,
            timestep=timestep,
            max_steps=max_steps,
            wall=wall,
            out=out,
            out_traj=out_traj,
            **args
        )


        self.update_mddynamics(velocities, temp, wall)
        self.update_mdstep(max_steps, out)
        self.add_trajectory_output(out_traj)
        self.update_time_step(timestep)

        self.update_parameters(self.context.parameters)
        

        self.context.calculate(
            symbols=image.symbols,
            positions=image.positions
        )
        
