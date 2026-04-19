#!/usr/bin/env python3
"""Exception hierarchy for deMonPy.

All errors raised by the package derive from :class:`DeMonError`, so user
code can catch the whole family with a single ``except DeMonError`` and
still discriminate by subclass when needed.
"""


class DeMonError(Exception):
    """Base class for every deMonPy-specific error."""


class ConfigError(DeMonError):
    """Raised when user-supplied parameters are missing or invalid."""


class InputError(DeMonError):
    """Raised when the ``deMon.inp`` file cannot be assembled."""


class ExecuteFailed(DeMonError):
    """Raised when the deMonNano binary returns a non-zero exit code."""


class OutputParseError(DeMonError):
    """Raised when ``deMon.out`` / ``deMon.mol`` cannot be parsed."""
