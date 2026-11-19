"""Warnings for unsupported parameter value types in the input writer.

Binary-free tests: the calculator is built without an executable and only
:meth:`~deMonPy.deMonNano.deMonNano.write_input` is called, so no deMon
binary is ever run.  A parameter value whose type the writer does not
know must emit a :class:`UserWarning` naming the parameter and the type
while the input file is still written (warning-only contract).  List
values are the documented exception: they are serialized by the section
writer that calls the handler (e.g. CUTSYS ``FRAGMENT``) and must not
warn.
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


def test_unknown_value_type_warns_and_still_writes_input(tmp_path):
    """An unsupported value type warns (naming the key and type) and writes."""
    calc = _calc(tmp_path, {"DFTB": {"SCC": True, "COLL": (10, 20)}})
    symbols, positions = ATOMS

    with pytest.warns(UserWarning, match=r"'COLL'.*tuple") as record:
        calc.write_input(symbols, positions)

    assert len(record) == 1  # a single warning, one per dropped value
    message = str(record[0].message)
    assert "tuple" in message
    assert "(10, 20)" in message
    # warning-only contract: the input file is still written
    assert (tmp_path / "deMon.inp").exists()
    assert "COLL" not in (tmp_path / "deMon.inp").read_text()  # value dropped


def test_supported_value_type_does_not_warn(tmp_path, recwarn):
    """The same structure with a supported value emits no UserWarning."""
    calc = _calc(tmp_path, {"DFTB": {"SCC": True, "COLL": 10}})
    symbols, positions = ATOMS

    calc.write_input(symbols, positions)

    assert not [w for w in recwarn.list if issubclass(w.category, UserWarning)]
    assert (tmp_path / "deMon.inp").exists()
    assert "COLL=10" in (tmp_path / "deMon.inp").read_text()


def test_list_value_does_not_warn(tmp_path, recwarn):
    """A list value (CUTSYS FRAGMENT) is consumed by the section writer and
    must not warn."""
    calc = _calc(tmp_path, {"CUTSYS": {"FRAGMENT": [3, 3], "RIGID": False}})
    symbols, positions = ATOMS

    calc.write_input(symbols, positions)

    assert not [w for w in recwarn.list if issubclass(w.category, UserWarning)]
    text = (tmp_path / "deMon.inp").read_text()
    assert "NMOL=2" in text
