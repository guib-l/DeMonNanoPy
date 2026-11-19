"""Auto-raise of deMon execution errors in ``deMonNano.calculate``.

Binary-free regression tests: a fake ``deMon.x`` shell script (same pattern
as the fake ``mpirun`` of :mod:`test_launcher`) stands in for the real
executable.  It writes a minimal ``deMon.out`` (and, for the clean run, a
valid ``deMon.mol``) and exits with a chosen status, so no real deMon
calculation is ever run.
"""

import stat

import pytest

from deMonPy.deMonNano import deMonNano
from deMonPy.exceptions import ExecuteFailed

PARAMS = {"DEMON_PARAMETERS": {"ACTIVE": {"DFTB": {"SCC": True}}}}

ATOMS = (["C", "C"], [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])


def _script(path, body):
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def _fake_exe(tmp_path, out_body, mol_body=None, exit_code=0):
    """Fake deMon.x: writes minimal deMon.out / deMon.mol, exits *exit_code*."""
    body = "cat > deMon.out <<'EOF'\n" + out_body + "EOF\n"
    if mol_body is not None:
        body += "cat > deMon.mol <<'EOF'\n" + mol_body + "EOF\n"
    body += f"exit {exit_code}\n"
    return _script(tmp_path / "deMon.x", body)


def _calc(exe, tmp_path):
    return deMonNano(
        execut=str(exe),
        workdir=str(tmp_path),
        basis={"PTYPE": "BIO", "SKFILE": "slako"},
        **PARAMS,
    )


def test_exit0_with_error_lines_raises(tmp_path):
    """An exit-0 run whose deMon.out holds an "ERROR :" line raises ExecuteFailed."""
    exe = _fake_exe(tmp_path, "ERROR : loop over charges not converged\n")
    calc = _calc(exe, tmp_path)
    symbols, positions = ATOMS
    with pytest.raises(ExecuteFailed) as excinfo:
        calc.calculate(symbols=symbols, positions=positions)
    message = str(excinfo.value)
    assert "loop over charges not converged" in message
    assert calc.title in message
    assert str(tmp_path) in message


def test_nonzero_exit_raises(tmp_path):
    """A non-zero exit status raises ExecuteFailed (already the case before the auto-raise)."""
    exe = _fake_exe(tmp_path, "", exit_code=3)
    calc = _calc(exe, tmp_path)
    symbols, positions = ATOMS
    with pytest.raises(ExecuteFailed, match="return code 3"):
        calc.calculate(symbols=symbols, positions=positions)


def test_clean_run_does_not_raise(tmp_path):
    """A successful exit-0 run without "ERROR :" lines returns normally."""
    mol_body = (
        " 2\n"
        " INPUT GEOMETRY\n"
        "  C      0.000000    0.000000    0.000000\n"
        "  C      1.400000    0.000000    0.000000\n"
    )
    exe = _fake_exe(tmp_path, "DFTB total energy = -1.0\n", mol_body=mol_body)
    calc = _calc(exe, tmp_path)
    symbols, positions = ATOMS
    calc.calculate(symbols=symbols, positions=positions)
    assert calc.has_errors() is False
    assert calc.results["energy"]["energy"] == pytest.approx(-1.0)


def test_timeout_raises_execute_failed(tmp_path):
    """A run exceeding the timeout raises ExecuteFailed mentioning the timeout.

    The fake executable sleeps far longer than the requested timeout so the
    test completes in ~1 s instead of waiting the full sleep.
    """
    exe = _script(tmp_path / "deMon.x", "sleep 30\n")
    calc = deMonNano(
        execut=str(exe),
        workdir=str(tmp_path),
        basis={"PTYPE": "BIO", "SKFILE": "slako"},
        timeout=1,
        **PARAMS,
    )
    symbols, positions = ATOMS
    with pytest.raises(ExecuteFailed, match="timeout") as excinfo:
        calc.calculate(symbols=symbols, positions=positions)
    message = str(excinfo.value)
    assert "timeout" in message
    assert "1" in message  # the requested timeout value
