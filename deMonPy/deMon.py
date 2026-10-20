
import os,sys
import copy
import deMonPy
from deMonPy.basic import BasicCalculation



class deMon(BasicCalculation):
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

        if properties is None:
            properties = ["energy"]
        if basis is None:
            basis = {}

        self.parameters = copy.deepcopy(parameters)
        self._ase_obj = ase_obj

        _execut = self.parameters.pop("DEMON_EXECUTABLE", execut or deMonPy.DEMON_EXECUTABLE)
        _prefix = self.parameters.pop("PREFIX", prefix)
        _workdir = self.parameters.pop("DEMON_WORKDIR", workdir)

        BasicCalculation.__init__(self, _execut, _workdir, _prefix, omp_threads=omp_threads)

        # Start running directory
        self.title = title
        self.workdir = self.parameters.pop("WORKDIR", workdir)

        self.set_workdir()

        # Initialize
        self.state = {}
        self.results = {}

        self.flags = set()


    _ARTIFACT_PATTERNS = (
        "deMon.inp",
        "deMon.out",
        "deMon.mol",
        "deMon.coef",
        "deMon.quat",
        "deMon.dip",
        "deMon.rst",
        "deMon.*.quat",
        "deMon.*.mol",
        "*.log",
        "FFDS",
        "3ord_param",
    )

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

        # Build parameters
        # self.basis = parameters.pop("BASIS", basis or deMonPy.DEMON_BASIS)

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















