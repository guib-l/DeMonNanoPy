
# Import standard de python3
import os
import re

import numpy as np

from deMonPy.exceptions import OutputParseError
from deMonPy.molden import read_XYZ
from deMonPy.profile import assert_flags, exclude_flags





class read_output:

    def __init__(self):
        ...

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







    