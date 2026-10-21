
import os,sys
import copy
import deMonPy
from deMonPy.basic import Common
from deMonPy.encoder import AseEncoder
from deMonPy.exceptions import ConfigError, ExecuteFailed
from deMonPy.demon.input import write_input
from deMonPy.demon.output import read_output


# ============================================================================= 
#
# MAIN object for deMon
#
# =============================================================================


class deMon(Common):
    """High-level calculator that builds input, runs deMon and parses output.

    A single :meth:`calculate` call writes the ``deMon.inp`` file, executes
    the binary, reads ``deMon.out`` / ``deMon.mol`` and stores the parsed
    results in :attr:`results`.

    The calculator is configured through nested dictionaries that mirror the
    deMonNano input-file structure.  Active parameter keys are automatically
    converted into lower-case *flags* that drive conditional I/O sections
    (see :func:`~deMonPy.profile.assert_flags`).

    Attributes:
        parameters: Snapshot of the keyword arguments passed at init time.
        title: Title string written to the generated input file.
        workdir: Directory where calculation files are read and written.
        basis: Basis / Slater-Koster file configuration dictionary.
        flags: ``set`` of lower-case flags derived from the active
            parameter and module keys.
        state: Dictionary of state snapshots produced by :meth:`calculate`.
        results: Dictionary of parsed results from the last calculation.
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

        
        Common.__init__(self, 
                        execut,
                        workdir,
                        omp_threads,
                        prefix,
                        title,
                        properties,
                        basis,
                        ase_obj,
                        clean_repository,
                        **parameters,
                    )


    _ARTIFACT_PATTERNS = (
        "deMon.inp",
        "deMon.out",
        "deMon.mol"
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

        if properties is None:
            properties = ["energy"]
        if basis is None:
            basis = {}

        self.parameters = parameters.copy()

        self._wi = write_input(BASIS=self.basis, **parameters)
        self.flags = self._wi.flags

        self._wo = read_output(
            properties=properties,
            workdir=self.workdir,
            flags=self.flags,
            output="deMon.out",
            ase_obj=self._ase_obj,
        )

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

    def write_input(self,):
        ...

    def parse_output(self,):
        ...















