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
from copy import deepcopy

import deMonPy
from deMonPy.profile import Process


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

    def __init__(self, executable, workdir, prefix, omp_threads=1):
        """Initialise the calculation process wrapper.

        Args:
            executable: Path to the deMonNano executable.
            workdir: Working directory used for input and output files.
            prefix: Prefix used by the process manager.
            omp_threads: Number of OpenMP threads to request.
                Defaults to ``1``.
        """
        self.process = Process(
            executable=executable,
            workdir=workdir,
            prefix=prefix,
            omp_threads=omp_threads,
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



