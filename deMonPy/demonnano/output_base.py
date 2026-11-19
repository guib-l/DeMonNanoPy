#!/usr/bin/env python3
"""Base class of the deMonNano output reader: file loading and parse helpers."""

import os

from deMonPy.exceptions import OutputParseError
from deMonPy.ioread import IOread


def convert_float(val, safe=False):
    try:
        return float(val)
    except ValueError:
        return val if not safe else None


def _is_int(token):
    try:
        int(token)
        return True
    except ValueError:
        return False



class OutputBase(IOread):
    """Shared state and parsing helpers of the deMonNano output readers."""

    def __init__(
        self,
        properties=None,
        workdir="./",
        output="deMon.out",
        flags=None,
        ase_obj=True,
        type_calculation=None,
        ignore_parse_errors=True,
    ):
        """Initialize the output reader.

        Args:
            properties: Requested properties to parse.
            workdir: Directory containing output files.
            output: Main output file name.
            flags: Active parser flags.
            type_calculation: Reserved calculation metadata.
            ignore_parse_errors: If ``True`` (default), a reader that
                fails records a ``"parse"`` error and leaves its results
                set to ``None`` instead of raising.  Missing or
                unreadable output files always raise.
        """

        IOread.__init__(
            self,
        )

        self.workdir = workdir
        self.output = output

        self.flags = set() if flags is None else flags
        self.ignore_parse_errors = ignore_parse_errors

        self.properties = ["energy"] if properties is None else properties
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
            with open(filename, "r", encoding="utf-8", errors="replace") as fd:
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

    # ------------------------------------------------------------------ guards

    def _check(self, ok, message, line=None):
        """Record a ``"parse"`` error with *message* when *ok* is false.

        Returns:
            bool: *ok*, so the call can be used in a condition.
        """
        if not ok:
            self._add_error("parse", message, line=None if line is None else line.strip())
        return ok

    def _parse_line(self, line, fmt, where, strict=True, record=True):
        """Convert the fields of *line* described by the ``LineFormat`` *fmt*.

        On a malformed line, a ``"parse"`` error naming *where* and *fmt* is
        recorded (unless ``record=False``), then the conversion error is
        re-raised (``strict=True``, caught by ``safe_parse`` as before) or
        ``None`` is returned (``strict=False``, the caller skips the line).

        Returns:
            dict | None: ``{key: converted value}`` for every field of *fmt*.
        """
        parts = line.split(fmt.sep)
        try:
            return {key: conv(parts[idx]) for key, (idx, conv) in fmt.fields.items()}
        except (ValueError, IndexError):
            self._check(
                not record,
                f"{where}: {fmt.name} expects fields like {fmt.sample!r}, "
                f"found {line.strip()!r}",
                line=line,
            )
            if strict:
                raise
            return None
