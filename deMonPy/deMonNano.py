#!/usr/bin/env python3
import __future__

# Import standard de python3
import os,sys
import numpy as np



import deMonPy
from deMonPy.profile import Process
from deMonPy.input import write_input
from deMonPy.output import read_output




from deMonPy.encoder import AseEncoder




class BasicCalculation:
    """Base class used to execute a deMonNano calculation."""
    
    execut  = ""
    workdir = None

    def __init__(
            self,
            exec,
            workdir,
            prefix,
            omp_threads=1,
            system=False):
        """Initialize the calculation process wrapper.

        Args:
            exec: Path to the deMonNano executable.
            workdir: Working directory used for input and output files.
            prefix: Prefix used by the process manager.
            omp_threads: Number of OpenMP threads to request.
            system: Whether the executable should be launched through the shell.
        """
        
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
        """Run the underlying process.

        Args:
            ignore_fails: If True, suppress execution exceptions.

        Raises:
            Exception: Propagated when execution fails and failures are not ignored.
        """

        try:
            self.process.execute()
        except Exception as e:
            if not ignore_fails:
                raise Exception(e)


    def set_workdir(self,):
        """Create the working directory if it does not already exist."""

        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)

    def set_state(self, index=1, **props):
        """Store a calculation state snapshot.

        Args:
            index: Identifier used to build the state key.
            **props: Properties to associate with the stored state.
        """
        self.state.update({"state-%s" % index: props})

    def get_state(self, index=1):
        """Return a previously stored state.

        Args:
            index: Identifier of the state to retrieve.

        Returns:
            dict: Stored state properties.
        """
        return self.state["state-%s" % index]
    
    def to_dict(self,):
        """Return the local namespace for serialization."""
        return locals()
    




class deMonNano(BasicCalculation):
    """High-level wrapper for preparing, running, and reading deMonNano jobs."""

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
        """Initialize a deMonNano calculator.

        Args:
            execut: Path to the deMonNano executable.
            workdir: Directory where calculation files are written.
            omp_threads: Number of OpenMP threads.
            system: Whether the executable is launched through the system shell.
            prefix: Prefix used by the process manager.
            title: Title written to the generated input.
            properies: Output properties requested from the parser.
            basis: Basis configuration passed to the input writer.
            **parameters: Additional input and runtime parameters.
        """
        
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
        """Clear stored states, results, and active flags."""
        
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
        """Refresh the input and output handlers with new parameters.

        Args:
            properies: Output properties requested from the parser.
            basis: Basis configuration passed to the input writer.
            **parameters: Additional input parameters.
        """
        
        self.parameters = parameters.copy()

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
        """Run a full calculation for a given structure.

        Args:
            symbols: Atomic symbols describing the system.
            positions: Atomic positions associated with the symbols.
            index: Identifier used to store the resulting state.
            **kwargs: Reserved for future calculation options.
        """
        
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
        """Write the deMonNano input files for the current calculation.

        Args:
            symbols: Atomic symbols describing the system.
            geometry: Atomic coordinates written to the geometry section.
        """
        
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
        """Parse deMonNano output files and update cached results."""

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
        is_md = False
        if "md" in self.flags:
            is_md = True
        self._wo.read_geometry(output='deMon.mol',
                               is_charges=True, 
                               velocities=is_md,
                               keep=1,)
        
        self.results = self._wo.complet_results

    def print_results(self, files=sys.stdout):
        """Serialize parsed results as formatted JSON.

        Args:
            files: Writable stream receiving the JSON output.
        """
        import json
        print(
            json.dumps(
                self._wo.complet_results, 
                indent=4, 
                ensure_ascii=True, 
                cls=AseEncoder
            ),
            file=files
        )



from deMonPy import available_modules


class Module_DeMonNano(deMonNano):
    """deMonNano calculator extended with pluggable runtime modules."""


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
        """Initialize a module-aware deMonNano calculator.

        Args:
            module: Name of the module to load.
            execut: Path to the deMonNano executable.
            workdir: Directory where calculation files are written.
            omp_threads: Number of OpenMP threads.
            system: Whether the executable is launched through the system shell.
            prefix: Prefix used by the process manager.
            title: Title written to the generated input.
            properies: Output properties requested from the parser.
            basis: Basis configuration passed to the input writer.
            available_modules: Registry of available module definitions.
            **parameters: Additional input and runtime parameters.
        """
        
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
        """Return the selected module definition."""
        return self._module

    @module.setter
    def module(self, module):
        """Set the active module from the available module registry.

        Args:
            module: Name of the module to activate.

        Raises:
            NotImplementedError: If the requested module is unknown.
        """
        if module not in available_modules.keys():
            raise NotImplementedError(f"{module} is not available")
        self._module = available_modules[module].copy()


    def initialize(self, **kwds):
        """Instantiate the configured module with the current context.

        Args:
            **kwds: Additional keyword arguments merged into the module options.
        """
        params = self.parameters
        module = self._module.pop("module", None)
        args   = self._module.pop("args", {})

        assert module is not None, ValueError("Unknow module")
        
        args.update(**kwds)
        params.update(**args)

        self.build = module(context=self, **params)
        self.is_build = True


    def reset(self):
        """Clear the built module instance."""
        self.is_build = False
        self.build    = None


    def __call__(self, method=None, **kwds):
        """Dispatch a call to the built module.

        Args:
            method: Optional module method name to invoke.
            **kwds: Keyword arguments forwarded to the module call.

        Returns:
            Any: Result returned by the module call.

        Raises:
            NotImplementedError: If the requested method does not exist.
        """
        
        if not self.is_build:
            self.initialize()

        if method is None:
            return self.build.__call__(**kwds)
        
        elif hasattr(self.build, method):
            func = getattr(self.build, method)
            return func(**kwds)
        
        else:
            raise NotImplementedError(
                f"Method {method} is unknow in module {self.build.__name__}"
            )






