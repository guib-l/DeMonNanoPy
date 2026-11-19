"""Geometry validation of ``deMonNano.calculate``.

Binary-free tests: invalid ``symbols``/``positions`` raise
:class:`~deMonPy.exceptions.InputError` before the input file is written,
so the failure cases need no executable at all.  The valid cases use the
fake ``deMon.x`` shell script pattern of :mod:`test_execute_errors` to
prove the run proceeds past validation.
"""

import stat

import numpy as np
import pytest

from deMonPy.deMonNano import deMonNano
from deMonPy.exceptions import InputError

PARAMS = {"DEMON_PARAMETERS": {"ACTIVE": {"DFTB": {"SCC": True}}}}

ATOMS = (["C", "C"], [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])


def _script(path, body):
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def _fake_exe(tmp_path, n_atoms=2):
    """Fake deMon.x: writes minimal deMon.out / deMon.mol, exits 0."""
    body = "cat > deMon.out <<'EOF'\nDFTB total energy = -1.0\nEOF\n"
    body += "cat > deMon.mol <<'EOF'\n"
    body += f" {n_atoms}\n"
    body += " INPUT GEOMETRY\n"
    body += "  C      0.000000    0.000000    0.000000\n"
    if n_atoms > 1:
        body += "  C      1.400000    0.000000    0.000000\n"
    body += "EOF\nexit 0\n"
    return _script(tmp_path / "deMon.x", body)


def _calc(tmp_path, n_atoms=2):
    """A :class:`~deMonPy.deMonNano.deMonNano` bound to the fake executable."""
    return deMonNano(
        execut=str(_fake_exe(tmp_path, n_atoms)),
        workdir=str(tmp_path),
        basis={"PTYPE": "BIO", "SKFILE": "slako"},
        **PARAMS,
    )


@pytest.mark.parametrize(
    "symbols, positions",
    [
        (["C", "C"], np.zeros((3, 3))),
        (["C", "C", "C"], np.zeros((2, 3))),
    ],
)
def test_mismatched_lengths_raise(tmp_path, symbols, positions):
    """Different numbers of symbols and position rows raise InputError."""
    calc = _calc(tmp_path)
    with pytest.raises(InputError) as excinfo:
        calc.calculate(symbols=symbols, positions=positions)
    message = str(excinfo.value)
    assert str(len(symbols)) in message
    assert str(positions.shape[0]) in message
    # validation raised before the input file was written
    assert not (tmp_path / "deMon.inp").exists()


def test_wrong_column_count_raises(tmp_path):
    """A (N, 2) positions array raises InputError naming the actual shape."""
    calc = _calc(tmp_path)
    symbols, _ = ATOMS
    with pytest.raises(InputError) as excinfo:
        calc.calculate(symbols=symbols, positions=np.zeros((2, 2)))
    assert "(2, 2)" in str(excinfo.value)


def test_one_d_positions_two_atoms_raise(tmp_path):
    """A 1-D (3,) positions array with two symbols raises InputError."""
    calc = _calc(tmp_path)
    symbols, _ = ATOMS
    with pytest.raises(InputError) as excinfo:
        calc.calculate(symbols=symbols, positions=np.array([0.0, 0.0, 0.0]))
    assert "(3,)" in str(excinfo.value)


def test_valid_two_atom_run_proceeds(tmp_path):
    """A valid (N, 3) geometry passes validation and the run completes."""
    calc = _calc(tmp_path)
    symbols, positions = ATOMS
    calc.calculate(symbols=symbols, positions=positions)
    assert calc.has_errors() is False
    assert calc.results["energy"]["energy"] == pytest.approx(-1.0)


def test_single_atom_one_row_run_proceeds(tmp_path):
    """A single atom given as (1, 3) -- the form the writer accepts -- passes."""
    calc = _calc(tmp_path, n_atoms=1)
    calc.calculate(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    assert calc.has_errors() is False
    assert calc.results["energy"]["energy"] == pytest.approx(-1.0)


def test_single_atom_one_d_raises(tmp_path):
    """A 1-D (3,) single-atom geometry is rejected (the writer rejects it too)."""
    calc = _calc(tmp_path, n_atoms=1)
    with pytest.raises(InputError, match=r"\(3,\)"):
        calc.calculate(symbols=["C"], positions=np.array([0.0, 0.0, 0.0]))
