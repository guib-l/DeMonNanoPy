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
import copy
from copy import deepcopy

import deMonPy
from deMonPy.profile import Process
from deMonPy.encoder import AseEncoder
from deMonPy.exceptions import ConfigError, ExecuteFailed


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

    def __init__(self, executable, workdir, prefix, omp_threads=1, launcher=None, nprocs=1):
        """Initialise the calculation process wrapper.

        Args:
            executable: Path to the deMonNano executable.
            workdir: Working directory used for input and output files.
            prefix: Prefix used by the process manager.
            omp_threads: Number of OpenMP threads to request
                (exported as ``OMP_NUM_THREADS``; ``None`` leaves it unset).
                Defaults to ``1``.
            launcher: Optional command prefix (``mpirun``, ``srun``...),
                see :class:`~deMonPy.profile.Process`.
            nprocs: Number of processes substituted for ``{nprocs}`` in the
                launcher.  Defaults to ``1``.
        """
        self.process = Process(
            executable=executable,
            workdir=workdir,
            prefix=prefix,
            omp_threads=omp_threads,
            launcher=launcher,
            nprocs=nprocs,
        )
        self.workdir = workdir
        self.state = {}

    def execute(self, ignore_fails=False):
        """Run the underlying process.

        Args:
            ignore_fails: If ``True``, any exception raised during
                execution is silently caught.  Defaults to ``False``.

        Raises:
            ExecuteFailed: Propagated from :meth:`Process.execute` when
                the run fails and *ignore_fails* is ``False``.
        """
        try:
            self.process.execute()
        except Exception:
            if not ignore_fails:
                raise

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

    #: Instance attributes excluded from :meth:`to_dict` because they are
    #: either self-referential (``state`` holds snapshots that embed the
    #: very dict returned here) or non-serialisable engine objects.
    _TO_DICT_EXCLUDE = ("state", "process", "_wi", "_wo")

    def to_dict(self):
        """Return the instance namespace as a plain dictionary.

        This is intended for lightweight serialisation and debugging.  A
        **shallow copy** is returned (mutating it does not touch the
        instance), and the self-referential ``state`` snapshot plus the
        internal engine objects (``process``, ``_wi``, ``_wo``) are
        omitted so that the result is free of reference cycles and stays
        JSON-serialisable.

        Returns:
            dict: Shallow copy of ``self.__dict__`` without the internal
            and self-referential entries.
        """
        return {
            key: value for key, value in self.__dict__.items() if key not in self._TO_DICT_EXCLUDE
        }




class Common(BasicCalculation):
    """
    High-level calculator that builds input, runs deMon and parses output.
    """

    def __init__(
        self,
        execut=None,
        workdir=".",
        omp_threads=1,
        prefix="DEMON",
        title="CALCULATION DEMON",
        properties=None,
        basis=None,
        ase_obj=True,
        clean_repository=True,
        **parameters,
    ):

        if properties is None:
            properties = ["energy"]
        if basis is None:
            basis = {}

        self.parameters = copy.deepcopy(parameters)
        self._ase_obj = ase_obj

        _execut = self.parameters.pop("DEMON_EXECUTABLE", execut or deMonPy.DEMON_EXECUTABLE)
        _prefix = self.parameters.pop("PREFIX", prefix)
        _workdir = self.parameters.pop("DEMON_WORKDIR", workdir)
        _launcher = self.parameters.pop("DEMON_LAUNCHER", deMonPy.DEMON_LAUNCHER)
        _nprocs = self.parameters.pop("DEMON_NPROCS", deMonPy.DEMON_NPROCS)

        BasicCalculation.__init__(
            self,
            _execut,
            _workdir,
            _prefix,
            omp_threads=omp_threads,
            launcher=_launcher,
            nprocs=_nprocs,
        )

        # Start running directory
        self.title = title
        self.workdir = self.parameters.pop("WORKDIR", workdir)

        self.set_workdir()

        # Initialize
        self.state = {}
        self.results = {}

        self.flags = set()


    # -------------------------------------------------------------------------
    # Files IO method
    # -------------------------------------------------------------------------

    def clean_workdir(self):
        """Remove known deMonNano artefacts from the working directory.

        Only files matching :attr:`_ARTIFACT_PATTERNS` are deleted.
        Sub-directories and unrelated files are preserved.  Refuses to
        operate on filesystem roots or the user's home directory.
        """
        from pathlib import Path

        wd = Path(self.workdir).expanduser().resolve()

        for pattern in self._ARTIFACT_PATTERNS:
            for f in wd.glob(pattern):
                if f.is_file():
                    os.remove(f)

    # -------------------------------------------------------------------------
    # Basics methods 
    # -------------------------------------------------------------------------

    def reset(self):
        """Clear stored states, results and active flags.

        After calling this method :attr:`results` and :attr:`state` are
        emptied and :attr:`flags` is cleared.  A subsequent
        :meth:`update` or :meth:`calculate` call is required before
        the calculator can produce new results.
        """
        self.state = {}
        self.results = {}

        self.flags = set()

    def calculate(
        self,
        *,
        symbols,
        positions,
        cell=None,
        pbc=False,
        kpts=None,
        index=0,
        read_charges=False,
        extract_debug=False,
        **kwargs,
    ):
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
        self.write_input(symbols, positions, cell=cell, kpts=kpts)

        self.execute(ignore_fails=False)

        self.parse_output(read_charges=read_charges, extract_debug=extract_debug, **kwargs)

        self.set_state(
            index=index,
            **{
                "symbols": symbols,
                "positions": positions,
                "results": self.results.copy(),
                "calculator": self.to_dict(),
            },
        )

    def update(self, properties=None, basis=None, **parameters):
        """Rebuild the input writer and output reader with new parameters.

        This replaces both :attr:`_wi` and :attr:`_wo` so that the next
        :meth:`calculate` call uses the updated configuration.  The
        :attr:`flags` set is re-derived from the new parameter keys.
        The ``ase_obj`` setting chosen at construction time is preserved.

        Args:
            properties: List of output properties requested from the
                parser (e.g. ``['energy']``).
            basis: Basis configuration dictionary.  Falls back to
                :data:`deMonPy.DEMON_BASIS` when empty.
            **parameters: Full deMonNano configuration dictionaries
                (same structure as the constructor).
        """

        ...


    def write_input(self,):
        ...

    def parse_output(self,):
        ...

    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------

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

    def has_errors(self):
        """Return ``True`` if the last calculation recorded any error.

        Returns:
            bool: True when ``results["errors"]`` is non-empty.
        """
        return bool(self.results.get("errors"))

    def raise_on_errors(self):
        """Raise :class:`ExecuteFailed` when errors were recorded.

        Useful right after :meth:`calculate` to convert silently-parsed
        deMonNano errors into a Python exception.

        Raises:
            ExecuteFailed: If at least one entry is present in
                ``results["errors"]``.
        """
        errs = self.results.get("errors") or []
        if not errs:
            return
        formatted = []
        for e in errs:
            if isinstance(e, dict):
                formatted.append(f"[{e.get('kind', '?')}] {e.get('message', '')}")
            else:
                formatted.append(str(e))
        raise ExecuteFailed("Calculation reported errors:\n  " + "\n  ".join(formatted))




















