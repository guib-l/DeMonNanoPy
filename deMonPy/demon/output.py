#!/usr/bin/env python3
import os

from deMonPy.exceptions import OutputParseError
from deMonPy.ioread import IOread


class read_output(IOread):
    """Parser for deMon output files."""

    def __init__(
        self,
        properties=["energy"],
        workdir="./",
        output="deMon.out",
        flags=set(),
        ase_obj=True,
        ignore_parse_errors=True,
    ):
        """Initialize the output reader.

        Args:
            properties: Requested properties to parse.
            workdir: Directory containing output files.
            output: Main output file name.
            flags: Active parser flags.
            ase_obj: Return geometries as ASE objects.
            ignore_parse_errors: If ``True`` (default), readers decorated
                with :func:`~deMonPy.ioread.safe_parse` record a
                ``"parse"`` error and leave their results set to ``None``
                instead of raising.  Missing or unreadable output files
                always raise.
        """
        IOread.__init__(self)

        self.workdir = workdir
        self.output = output
        self.flags = flags
        self.properties = properties
        self.ignore_parse_errors = ignore_parse_errors
        self.complet_results = {
            "errors": [],
        }
        self._ase_obj = ase_obj
        self.lines = []

    def read_file(
        self,
    ):
        """Load the main output file into memory.

        Raises:
            OutputParseError: If the output file is missing or
                unreadable (typical when the binary crashed before
                writing anything).
        """

        filename = os.path.join(self.workdir, self.output)
        try:
            with open(filename, "r") as fd:
                self.lines.extend(fd.readlines())
        except FileNotFoundError as err:
            raise OutputParseError(
                f"deMon output not found at {filename}. "
                f"Did the calculation fail before writing output?"
            ) from err
        except OSError as err:
            raise OutputParseError(f"Cannot read deMon output at {filename}: {err}") from err

    def _add_error(self, kind, message, line=None):
        """Append a structured error entry to the results.

        Args:
            kind: Short error category (e.g. ``"mpi"``, ``"tddftb"``,
                ``"scf"``, ``"optimization"``).
            message: Human-readable description.
            line: Optional source line the error was extracted from.
        """
        self.complet_results.setdefault("errors", []).append(
            {
                "kind": kind,
                "message": message,
                "line": line,
            }
        )







    