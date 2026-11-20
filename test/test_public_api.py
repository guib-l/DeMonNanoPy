"""Top-level public API re-exports in ``deMonPy/__init__.py``.

Pure namespace checks: every re-exported name must be reachable at the
package top level and be the *same* object as the one defined in its home
module (identity proves re-export, not redefinition). No binary, no
``global.json``, no network, no ``calculate()`` calls.

Also pins the deep-import pattern (``from deMonPy.deMonNano import
deMonNano``) that existing examples and tests rely on: it resolves against
the submodule in ``sys.modules`` and must keep working even though the
package attribute ``deMonPy.deMonNano`` is now shadowed by the engine
class.
"""

import inspect
import sys

import deMonPy
from deMonPy.ase_calculator import DeMonNano as _ase
from deMonPy.deMonNano import deMonNano as _deep
from deMonPy.exceptions import (
    ConfigError as _ConfigError,
    DeMonError as _DeMonError,
    ExecuteFailed as _ExecuteFailed,
    InputError as _InputError,
    OutputParseError as _OutputParseError,
)

# ``import deMonPy.deMonNano as X`` would bind the shadowed class, so the
# defining module is fetched from ``sys.modules`` instead.
_eng_mod = sys.modules["deMonPy.deMonNano"]


def test_engine_class_reexported_identity():
    assert inspect.isclass(deMonPy.deMonNano)
    assert callable(deMonPy.deMonNano)
    assert deMonPy.deMonNano is _eng_mod.deMonNano


def test_engine_workflow_reexported_identity():
    assert deMonPy.Module_DeMonNano is _eng_mod.Module_DeMonNano


def test_ase_calculator_reexported_identity():
    assert deMonPy.DeMonNano is _ase


def test_exceptions_reexported_identity():
    assert deMonPy.DeMonError is _DeMonError
    assert deMonPy.ConfigError is _ConfigError
    assert deMonPy.InputError is _InputError
    assert deMonPy.ExecuteFailed is _ExecuteFailed
    assert deMonPy.OutputParseError is _OutputParseError


def test_exceptions_inherit_from_base():
    assert issubclass(deMonPy.ConfigError, deMonPy.DeMonError)
    assert issubclass(deMonPy.InputError, deMonPy.DeMonError)
    assert issubclass(deMonPy.ExecuteFailed, deMonPy.DeMonError)
    assert issubclass(deMonPy.OutputParseError, deMonPy.DeMonError)


def test_deep_import_still_works_after_shadowing():
    # from-import resolves via sys.modules, not the (now class) package
    # attribute, so it must yield the same class as the top-level name.
    assert _deep is deMonPy.deMonNano
