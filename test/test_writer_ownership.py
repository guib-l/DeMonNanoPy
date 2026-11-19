"""Ownership of the parameter snapshot taken by ``write_input.__init__``.

Binary-free tests: the calculator is built without an executable and only
:meth:`~deMonPy.deMonNano.deMonNano.write_input` is called, so no deMon
binary is ever run.  The input writer must deep-copy the ACTIVE blocks
(``DEMON_PARAMETERS.ACTIVE`` and ``DEMON_MODULE.ACTIVE``) at
construction: its in-place section consumption (``pop``/``update``) must
stay private to the writer, so the calculator's ``self.parameters``
snapshot and the caller's original dictionaries survive a full write, and
two consecutive ``write_input`` calls on the same calculator emit the
identical ``deMon.inp`` (the step-2 input keeps every section the step-1
input had).
"""

import pytest

from deMonPy.deMonNano import deMonNano

ATOMS = (["C", "C"], [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])

ACTIVE = {"DFTB": {"SCC": True, "NOR": 1.0}, "CI": {"CONST": None}}
MODULE = {"OPT": {"TRAJECTORY": True, "SP": True}}


def _calc(tmp_path, parameters, module):
    """A :class:`~deMonPy.deMonNano.deMonNano` bound to no executable."""
    return deMonNano(
        execut=None,
        workdir=str(tmp_path),
        basis={"PTYPE": "BIO", "SKFILE": "slako"},
        DEMON_PARAMETERS={"ACTIVE": parameters},
        DEMON_MODULE={"ACTIVE": module},
    )


def test_writer_works_on_private_copies(tmp_path):
    """The writer's ACTIVE/MODULE dicts are independent of every caller."""
    calc = _calc(tmp_path, dict(ACTIVE), dict(MODULE))

    active = calc.parameters["DEMON_PARAMETERS"]["ACTIVE"]
    module = calc.parameters["DEMON_MODULE"]["ACTIVE"]
    assert calc._wi.parameters is not active  # no reference sharing
    assert calc._wi.parameters["DFTB"] is not active["DFTB"]
    assert calc._wi.module is not module
    assert calc._wi.module["OPT"] is not module["OPT"]


def test_write_input_twice_is_stable_and_keeps_snapshots(tmp_path):
    """Two writes are byte-identical and no owned dict is consumed."""
    active = {"DFTB": {"SCC": True, "NOR": 1.0}, "CI": {"CONST": None}}
    module = {"OPT": {"TRAJECTORY": True, "SP": True}}
    calc = _calc(tmp_path, active, module)
    symbols, positions = ATOMS

    # (a) first and second write_input calls emit byte-identical inputs.
    calc.write_input(symbols, positions)
    first = (tmp_path / "deMon.inp").read_bytes()

    calc.write_input(symbols, positions)
    second = (tmp_path / "deMon.inp").read_bytes()

    assert first == second

    # The step-2 file concretely contains every section the step-1 file
    # had (the writer pops CI.CONST and OPT.TRAJECTORY while writing).
    text = second.decode("utf-8")
    assert "DFTB" in text
    assert "SCC" in text
    assert "NOR=1.0" in text
    assert "CI" in text  # CI section marker (CONST=None -> bare keyword)
    assert "OPTIMIZATION" in text
    assert "SP" in text
    assert "GEOMETRY" in text

    # (b) the calculator's own snapshot still holds the full ACTIVE
    # structure: the writer's in-place pops must not have reached it.
    assert calc.parameters["DEMON_PARAMETERS"]["ACTIVE"] == {
        "DFTB": {"SCC": True, "NOR": 1.0},
        "CI": {"CONST": None},
    }
    assert calc.parameters["DEMON_MODULE"]["ACTIVE"] == {
        "OPT": {"TRAJECTORY": True, "SP": True}
    }

    # (c) the caller's original dictionaries (passed at construction) are
    # intact as well (regression guard for the update()/init copy
    # boundary).
    assert active == {"DFTB": {"SCC": True, "NOR": 1.0}, "CI": {"CONST": None}}
    assert module == {"OPT": {"TRAJECTORY": True, "SP": True}}


def test_write_is_reentrant_and_keeps_geometry_last(tmp_path):
    """A second direct ``write()`` is byte-identical and GEOMETRY stays last."""
    calc = _calc(tmp_path, dict(ACTIVE), dict(MODULE))
    symbols, positions = ATOMS

    calc.write_input(symbols, positions)
    first = (tmp_path / "deMon.inp").read_bytes()

    # (a) a second direct write() call must not consume the GEOMETRY entry.
    calc._wi.write(workdir=str(tmp_path))
    second = (tmp_path / "deMon.inp").read_bytes()

    # (b) the second write is byte-identical to the first.
    assert second == first

    # (c) the GEOMETRY entry still exists in the writer state after both.
    assert "GEOMETRY" in calc._wi.io_lines

    # (d) the geometry block is still at the END of the file: the whole
    # GEOMETRY section (header + one line per atom) is the trailing
    # region, and the file ends with the last atom line of that block.
    geom_block = calc._wi.io_lines["GEOMETRY"][-1]
    assert geom_block.startswith("GEOMETRY ")
    text = second.decode("utf-8")
    assert text.endswith(geom_block)
    assert text.splitlines()[-1] == geom_block.splitlines()[-1]


def test_paths_multi_checkpoint_warns_but_still_writes(tmp_path):
    """NPTS>1: a UserWarning fires, but the PATHS section is still written.

    The ACTIVE key ``PATHS`` sets the ``paths`` flag, so the
    ``@assert_flags("paths")``-guarded ``_write_paths`` runs.  Multi-path
    (NPTS > 1) is unsupported: the writer must emit a
    :class:`UserWarning` (never a stdout print) while still writing the
    PATHS section unchanged.
    """
    active = {"PATHS": {"NPTS": 2, "NPTHS": 1, "CHKPTS": {"PATH": [[0.0, 0.0, 0.0]]}}}
    calc = _calc(tmp_path, active, {})
    symbols, positions = ATOMS

    with pytest.warns(UserWarning, match="Multi-path input is not supported"):
        calc.write_input(symbols, positions)

    text = (tmp_path / "deMon.inp").read_text(encoding="utf-8")
    assert "NPTS=2" in text
    assert "NPTHS=1" in text


def test_paths_single_checkpoint_is_silent(tmp_path, recwarn):
    """NPTS=1: the PATHS section is written and no UserWarning is emitted."""
    active = {"PATHS": {"NPTS": 1, "NPTHS": 1, "CHKPTS": {"PATH": [[0.0, 0.0, 0.0]]}}}
    calc = _calc(tmp_path, active, {})
    symbols, positions = ATOMS

    calc.write_input(symbols, positions)

    text = (tmp_path / "deMon.inp").read_text(encoding="utf-8")
    assert "NPTS=1" in text
    assert "NPTHS=1" in text

    user_warnings = [w for w in recwarn.list if issubclass(w.category, UserWarning)]
    assert user_warnings == []
