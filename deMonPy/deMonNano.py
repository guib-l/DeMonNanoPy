#!/usr/bin/env python3
"""High-level interface for preparing, running and reading deMonNano jobs.

This module exposes three classes organised in an inheritance chain:

* :class:`BasicCalculation` -- thin wrapper around :class:`deMonPy.profile.Process`
  that handles working-directory creation, execution and state snapshots.
* :class:`deMonNano` -- full calculator that builds an input file, runs the
  binary and parses the output in a single :meth:`~deMonNano.calculate` call.
* :class:`Module_DeMonNano` -- extends :class:`deMonNano` with pluggable
  workflow modules (optimisation, MD, PTMC, ...).

Typical usage::

    from deMonPy.deMonNano import deMonNano

    calc = deMonNano(
        execut="/path/to/deMon.x",
        workdir="./run",
        basis={"PTYPE": "MAT", "SKFILE": "/path/to/slako"},
        DEMON_PARAMETERS={"ACTIVE": {"DFTB": {"SCC": True}}},
    )
    calc.calculate(symbols=["C", "C"], positions=[[0, 0, 0], [1.4, 0, 0]])
    print(calc.results)
"""

import os
import sys
import glob

import numpy as np

import deMonPy
from deMonPy.profile import Process
from deMonPy.input import write_input
from deMonPy.output import read_output
from deMonPy.encoder import AseEncoder




class BasicCalculation:
    """Base class that wraps a single deMonNano process execution.

    This class owns a :class:`~deMonPy.profile.Process` instance and
    provides helpers shared by every calculation type: working-directory
    creation, execution with optional error suppression, and a simple
    state-snapshot mechanism.

    Attributes:
        execut: Default executable path (class-level fallback).
        workdir: Working directory for input / output files.
        process: The :class:`~deMonPy.profile.Process` used to run
            the deMonNano binary.
    """

    execut = ""
    workdir = None

    def __init__(self, exec, workdir, prefix, omp_threads=1, system=False):
        """Initialise the calculation process wrapper.

        Args:
            exec: Path to the deMonNano executable.
            workdir: Working directory used for input and output files.
            prefix: Prefix used by the process manager.
            omp_threads: Number of OpenMP threads to request.
                Defaults to ``1``.
            system: If ``True`` the executable is launched through
                :func:`os.system`; otherwise :mod:`subprocess` is used.
                Defaults to ``False``.
        """
        self.process = Process(
            executable=exec,
            workdir=workdir,
            prefix=prefix,
            omp_threads=omp_threads,
            system=system,
        )

    def execute(self, ignore_fails=False):
        """Run the underlying process.

        Args:
            ignore_fails: If ``True``, any exception raised during
                execution is silently caught.  Defaults to ``False``.

        Raises:
            Exception: Propagated from :meth:`Process.execute` when
                the run fails and *ignore_fails* is ``False``.
        """
        try:
            self.process.execute()
        except Exception as e:
            if not ignore_fails:
                raise Exception(e)

    def set_workdir(self):
        """Create the working directory if it does not already exist."""
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)

    def set_state(self, index=1, **props):
        """Store a calculation state snapshot.

        Each snapshot is kept under the key ``"state-<index>"`` inside
        :attr:`state`.

        Args:
            index: Identifier used to build the state key.
            **props: Arbitrary properties to associate with this state.
        """
        self.state.update({"state-%s" % index: props})

    def get_state(self, index=1):
        """Retrieve a previously stored state snapshot.

        Args:
            index: Identifier of the state to retrieve.

        Returns:
            dict: The properties that were saved for the requested state.

        Raises:
            KeyError: If no state with the given *index* exists.
        """
        return self.state["state-%s" % index]

    def to_dict(self):
        """Return the instance namespace as a plain dictionary.

        This is intended for lightweight serialisation and debugging.

        Returns:
            dict: Shallow copy of ``self.__dict__``.
        """
        return self.__dict__
    




class deMonNano(BasicCalculation):
    """High-level calculator that builds input, runs deMonNano and parses output.

    A single :meth:`calculate` call writes the ``deMon.inp`` file, executes
    the binary, reads ``deMon.out`` / ``deMon.mol`` and stores the parsed
    results in :attr:`results`.

    The calculator is configured through nested dictionaries that mirror the
    deMonNano input-file structure.  Active parameter keys are automatically
    converted into lower-case *flags* that drive conditional I/O sections
    (see :func:`~deMonPy.profile.assert_flags`).

    Attributes:
        available_properties: Property names that can be populated by the
            output parser (currently ``["energies", "forces"]``).
        parameters: Snapshot of the keyword arguments passed at init time.
        title: Title string written to the generated input file.
        workdir: Directory where calculation files are read and written.
        basis: Basis / Slater-Koster file configuration dictionary.
        flags: ``set`` of lower-case flags derived from the active
            parameter and module keys.
        state: Dictionary of state snapshots produced by :meth:`calculate`.
        results: Dictionary of parsed results from the last calculation.
    """

    available_properties = ["energies", "forces"]

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
        """Initialise a deMonNano calculator.

        Args:
            execut: Path to the deMonNano executable.  Falls back to
                :data:`deMonPy.DEMON_EXECUTABLE` when ``None``.
            workdir: Directory where calculation files are written.
                Defaults to the current directory.
            omp_threads: Number of OpenMP threads.  Defaults to ``1``.
            system: If ``True`` (default) the executable is launched
                through :func:`os.system`; otherwise :mod:`subprocess`
                is used.
            prefix: Prefix used by the process manager.
                Defaults to ``"DEMON"``.
            title: Title written to the generated input.
                Defaults to ``"CALCULATION DEMONANO"``.
            properies: List of output properties requested from the
                parser (e.g. ``['energy']``).

                .. note:: The parameter name is intentionally kept as
                   ``properies`` for backward compatibility.

            basis: Basis configuration dictionary.  Expected keys are
                ``"PTYPE"`` and ``"SKFILE"``.  Falls back to
                :data:`deMonPy.DEMON_BASIS` when empty.
            **parameters: Full deMonNano configuration.  Recognised
                top-level keys include ``DEMON_EXECUTABLE``,
                ``DEMON_WORKDIR``, ``DEMON_PARAMETERS`` and
                ``DEMON_MODULE``.
        """
        
        self.parameters = parameters.copy()
        
        _execut  = parameters.pop("DEMON_EXECUTABLE",execut or deMonPy.DEMON_EXECUTABLE)
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
        self.basis = parameters.pop("BASIS",basis or deMonPy.DEMON_BASIS)

        self._wi = write_input(BASIS=self.basis,
                               **parameters)
        self.flags = self._wi.flags
        
        self._wo = read_output(properties=properies,
                               workdir=self.workdir,
                               flags=self.flags,
                               output="deMon.out")
        
    def clean_workdir(self):
        """Remove every file inside the working directory.

        .. warning::

            This deletes **all** files matched by ``<workdir>/*``
            without confirmation.  It does not recurse into
            sub-directories.
        """
        files = glob.glob(os.path.join(self.workdir, "*"))
        for f in files:
            os.remove(f)
                
    def reset(self):
        """Clear stored states, results and active flags.

        After calling this method every entry in :attr:`results` is
        reset to ``None`` and :attr:`flags` is emptied.  A subsequent
        :meth:`update` or :meth:`calculate` call is required before
        the calculator can produce new results.
        """
        self.state = {}
        self.results = {}

        self.flags = set()

        for props in self.available_properties:
            self.results.update({props: None})

    def update(
            self,
            properies=['energy'],
            basis={},
            **parameters):
        """Rebuild the input writer and output reader with new parameters.

        This replaces both :attr:`_wi` and :attr:`_wo` so that the next
        :meth:`calculate` call uses the updated configuration.  The
        :attr:`flags` set is re-derived from the new parameter keys.

        Args:
            properies: List of output properties requested from the
                parser (e.g. ``['energy']``).
            basis: Basis configuration dictionary.  Falls back to
                :data:`deMonPy.DEMON_BASIS` when empty.
            **parameters: Full deMonNano configuration dictionaries
                (same structure as the constructor).
        """
        
        self.parameters = parameters.copy()

        # Build parameters
        self.basis = parameters.pop("BASIS",basis or deMonPy.DEMON_BASIS)

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
            read_charges=False,
            extract_debug=False,
            **kwargs):
        """Run a full single-point (or flagged) calculation.

        This is the main entry point.  It sequentially:

        1. Writes the ``deMon.inp`` file via :meth:`write_input`.
        2. Executes the binary via :meth:`execute`.
        3. Parses ``deMon.out`` / ``deMon.mol`` via :meth:`read_output`.
        4. Stores a state snapshot accessible through :meth:`get_state`.

        Args:
            symbols: Sequence of atomic symbols (e.g. ``["C", "O"]``).
                Must match the length of *positions*.
            positions: Array-like of shape ``(N, 3)`` with Cartesian
                coordinates in angstroms.
            index: Integer identifier used to key the resulting state
                snapshot.  Defaults to ``0``.
            **kwargs: Reserved for future calculation options.
        """
        
        self.write_input(
            symbols,
            positions,)
        
        self.execute(ignore_fails=False)


        self.read_output(read_charges=read_charges, 
                         extract_debug=extract_debug)

        
        self.set_state(
            index=index,
            **{
                "symbols":symbols,
                "positions":positions,
                "results":self.results.copy(),
                "calculator":self.to_dict()
            }
        )

    def write_input(self, symbols, geometry):
        """Assemble and write the ``deMon.inp`` file.

        Every parameter and module section guarded by
        :func:`~deMonPy.profile.assert_flags` is called in turn;
        sections whose flag is absent are silently skipped.  The
        geometry block is always written last.

        The resulting file is saved to ``<workdir>/deMon.inp``.

        Args:
            symbols: Sequence of atomic symbols (e.g. ``["C", "O"]``).
            geometry: Array-like of shape ``(N, 3)`` with Cartesian
                coordinates written to the ``GEOMETRY`` section.
        """

        self.clean_workdir()

        # Parameters
        self._wi._write_dftb()
        self._wi._write_charge()
        self._wi._write_bondparam_wmull(symbols)
        self._wi._write_bondparam_cm3(symbols)
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

        # Geometry writing
        self._wi._write_geometry(symbols=symbols,
                                 positions=geometry)

        self._wi.write(workdir=self.workdir)


    def read_output(self, read_charges=False, extract_debug=False):
        """Parse deMonNano output files and populate :attr:`results`.

        The method loads ``deMon.out`` into memory, then calls every
        flag-guarded reader in sequence (energies, CI, TD-DFTB,
        frequencies, module-specific sections, geometry, errors).

        When the ``"md"`` flag is active, the geometry reader also
        extracts charges and velocities from ``deMon.mol``.

        After completion :attr:`results` mirrors the full
        :attr:`~deMonPy.output.read_output.complet_results` dictionary.
        """
        is_md = False
        is_charges = read_charges

        # Parameters
        self._wo.read_file()
        self._wo.read_basics()
        self._wo.read_freq()

        self._wo.read_energy()
        self._wo.read_ci()
        self._wo.read_tddftb()

        self._wo.read_print()

        # Modules
        self._wo._read_opt()
        self._wo._read_ptmc()
        self._wo._read_md()
        self._wo._read_neb()
        self._wo.parse_tensors()

        self._wo.read_debug(extract_data=extract_debug)

        # Geometry reading
        if "md" in self.flags:
            is_charges = True
            is_md = True

        self._wo.read_geometry(output='deMon.mol',
                               is_charges=is_charges,
                               velocities=is_md,
                               keep=1)

        self._wo.read_errors()
        self.results = self._wo.complet_results

    def print_results(self, files=sys.stdout):
        """Pretty-print parsed results as indented JSON.

        The output is encoded with :class:`~deMonPy.encoder.AseEncoder`
        so that NumPy arrays and ASE objects are serialisable.

        Args:
            files: Writable text stream that receives the JSON output.
                Defaults to :data:`sys.stdout`.
        """
        import json
        print(
            json.dumps(
                self._wo.complet_results,
                indent=4,
                ensure_ascii=True,
                cls=AseEncoder,
            ),
            file=files,
        )



from deMonPy import available_modules


class Module_DeMonNano(deMonNano):
    """Calculator extended with pluggable high-level workflow modules.

    This class adds a *module* layer on top of :class:`deMonNano`.
    Instead of running a single ``calculate()`` call the user selects a
    workflow module (optimisation, molecular dynamics, PTMC, ...) and
    dispatches work to it through :meth:`__call__`.

    The available modules are registered in
    :data:`deMonPy.available_modules`.

    Lifecycle::

        calc = Module_DeMonNano(module="md", execut="...", basis={...}, ...)
        calc.initialize(image=atoms)     # build the module instance
        calc(image=atoms, max_steps=200) # run the workflow

    Attributes:
        module: The currently selected module definition (a dictionary
            copied from :data:`~deMonPy.available_modules`).
        is_build: ``True`` once :meth:`initialize` has been called
            successfully.
        build: The instantiated module object (a subclass of
            :class:`~deMonPy.modules.module.modules`), or ``None``
            before initialisation.
    """

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
        """Initialise a module-aware deMonNano calculator.

        Args:
            module: Name of the module to activate.  Must be a key
                present in *available_modules* (e.g. ``"opt"``,
                ``"md"``, ``"ptmc"``).
            execut: Path to the deMonNano executable.  Falls back to
                :data:`deMonPy.DEMON_EXECUTABLE` when ``None``.
            workdir: Directory where calculation files are written.
                Defaults to the current directory.
            omp_threads: Number of OpenMP threads.  Defaults to ``1``.
            system: If ``True`` (default) the executable is launched
                through :func:`os.system`.
            prefix: Prefix used by the process manager.
                Defaults to ``"DEMON"``.
            title: Title written to the generated input.
                Defaults to ``"CALCULATION MODULE-DEMONANO"``.
            properies: List of output properties requested from the
                parser.
            basis: Basis configuration dictionary.
            available_modules: Registry mapping module names to their
                definition dictionaries.  Defaults to
                :data:`deMonPy.available_modules`.
            **parameters: Additional deMonNano configuration passed
                through to :class:`deMonNano`.
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
        """dict: The currently selected module definition dictionary."""
        return self._module

    @module.setter
    def module(self, module):
        """Set the active module by name.

        The *module* string is looked up in
        :data:`~deMonPy.available_modules` and a **shallow copy** of the
        matching definition is stored so that the registry entry is not
        mutated.

        Args:
            module: Name of the module to activate (e.g. ``"opt"``).

        Raises:
            NotImplementedError: If *module* is not a registered name.
        """
        if module not in available_modules.keys():
            raise NotImplementedError(f"{module} is not available")
        self._module = available_modules[module].copy()


    def initialize(self, **kwds):
        """Instantiate the selected module and bind it to this calculator.

        The module class stored under the ``"module"`` key of
        :attr:`module` is instantiated with ``context=self`` and the
        merged parameter dictionaries.  After this call :attr:`is_build`
        is ``True`` and :attr:`build` holds the live module instance.

        Args:
            **kwds: Extra keyword arguments merged into the module's
                default argument dictionary before instantiation.

        Raises:
            AssertionError: If the module definition has no ``"module"``
                class entry.
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
        """Clear the built module instance.

        After calling this method :attr:`is_build` is ``False`` and
        :attr:`build` is ``None``.  A new :meth:`initialize` (or an
        implicit one through :meth:`__call__`) is required before the
        calculator can dispatch work again.
        """
        self.is_build = False
        self.build = None


    def __call__(self, method=None, **kwds):
        """Dispatch a call to the active module.

        If the module has not been initialised yet, :meth:`initialize`
        is called automatically.

        When *method* is ``None`` the module's own ``__call__`` (i.e.
        its :meth:`~deMonPy.modules.module.modules.forward` entry
        point) is invoked.  Otherwise the named method is looked up on
        the module instance.

        Args:
            method: Optional name of a specific method to invoke on the
                module.  When ``None`` (default) the module's
                ``__call__`` is used.
            **kwds: Keyword arguments forwarded to the module method.

        Returns:
            The value returned by the module method.

        Raises:
            NotImplementedError: If *method* is given but does not exist
                on the built module.
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






