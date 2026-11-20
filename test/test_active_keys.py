"""Warning for ACTIVE keys without a matching input writer.

Binary-free tests: the calculator is built without an executable and only
:meth:`~deMonPy.deMonNano.deMonNano.write_input` is called, so no deMon
binary is ever run.  An ACTIVE key with no writer must emit a single
:class:`UserWarning` naming the key and the supported keys while the
input file is still written (warning-only contract).
"""

import pytest

from deMonPy.deMonNano import deMonNano

ATOMS = (["C", "C"], [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])


def _calc(tmp_path, active):
    """A :class:`~deMonPy.deMonNano.deMonNano` bound to no executable."""
    return deMonNano(
        execut=None,
        workdir=str(tmp_path),
        basis={"PTYPE": "BIO", "SKFILE": "slako"},
        DEMON_PARAMETERS={"ACTIVE": active},
    )


def test_valid_active_keys_do_not_warn(tmp_path, recwarn):
    """A valid ACTIVE dict (as in test_execute_errors) emits no UserWarning."""
    calc = _calc(tmp_path, {"DFTB": {"SCC": True}})
    symbols, positions = ATOMS

    calc.write_input(symbols, positions)

    assert not [w for w in recwarn.list if issubclass(w.category, UserWarning)]
    assert (tmp_path / "deMon.inp").exists()
