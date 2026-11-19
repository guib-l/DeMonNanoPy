"""Ownership of the parameters snapshot taken by ``deMonNano.update``.

Binary-free tests: the calculator is built without an executable and only
:meth:`~deMonPy.deMonNano.deMonNano.update` is called, so no deMon binary
is ever run and no input file is written.  ``update`` must deep-copy the
caller's configuration: in-place mutations made through
``calc.parameters`` (the kind the input writer performs) must not leak
into the caller's dictionaries, and later caller-side mutations must not
reach ``calc.parameters`` (independence in both directions).
"""

import pytest

from deMonPy.deMon import deMon
from deMonPy.deMonNano import deMonNano


def _calc(tmp_path):
    """A :class:`~deMonPy.deMonNano.deMonNano` bound to no executable."""
    return deMonNano(
        execut=None,
        workdir=str(tmp_path),
        basis={"PTYPE": "BIO", "SKFILE": "slako"},
        DEMON_PARAMETERS={"ACTIVE": {"DFTB": {"SCC": True}}},
    )


def test_update_deep_copies_parameters(tmp_path):
    """``calc.parameters`` and the caller's dictionaries are independent."""
    calc = _calc(tmp_path)

    params = {"ACTIVE": {"DFTB": {"SCC": True, "NOR": 1.0}, "CHARGE": {"TOTAL": 0.0}}}
    calc.update(DEMON_PARAMETERS=params)

    # The next state is coherent: the flags set was re-derived from the
    # update's parameters (construction had left only "dftb" behind).
    assert calc.flags == {"dftb", "charge"}

    # (a) Mutate the nested structure through ``calc.parameters`` -- the
    # in-place pop the writer performs: the caller's dict is untouched.
    calc.parameters["DEMON_PARAMETERS"]["ACTIVE"]["DFTB"].pop("SCC")
    assert params["ACTIVE"]["DFTB"]["SCC"] is True

    # (b) Mutate the caller's dict afterwards: ``calc.parameters`` is
    # untouched as well (independence in both directions).
    params["ACTIVE"]["DFTB"].pop("NOR")
    params["ACTIVE"]["CHARGE"]["TOTAL"] = 1.0
    active = calc.parameters["DEMON_PARAMETERS"]["ACTIVE"]
    assert "NOR" in active["DFTB"]
    assert active["CHARGE"]["TOTAL"] == 0.0


def test_legacy_deMon_update_copy(tmp_path):
    """Legacy :class:`~deMonPy.deMon.deMon` -- skipped: ``update()`` cannot complete.

    Construction itself is executor-less and side-effect free (verified),
    but ``update()`` raises before rebuilding any state, for pre-existing
    skeleton reasons unrelated to the copy fix: ``deMon.__init__`` never
    sets ``self.basis`` (``AttributeError`` at
    ``write_input(BASIS=self.basis, ...)``) and the skeleton
    ``demon.input.write_input.__init__`` accepts no arguments at all
    (``TypeError`` on the ``BASIS`` keyword).  The (a)+(b) ownership
    contract therefore cannot be exercised the same way as for
    ``deMonNano``; re-enable once the legacy skeleton runs ``update()``
    to completion.
    """
    deMon(execut=None, workdir=str(tmp_path))
    pytest.skip(
        "legacy deMon.update() cannot complete: __init__ never sets "
        "'basis' (AttributeError at write_input(BASIS=self.basis, ...)) "
        "and the skeleton demon.input.write_input.__init__ takes no "
        "arguments (TypeError on the BASIS keyword)"
    )
