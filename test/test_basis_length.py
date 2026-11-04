"""Basis lines longer than 72 characters must be rejected at configuration."""

import pytest

from deMonPy.deMonNano import deMonNano
from deMonPy.demonnano.input import MAX_BASIS_LENGTH, write_input
from deMonPy.exceptions import ConfigError

PARAMS = {"DEMON_PARAMETERS": {"ACTIVE": {"DFTB": {"SCC": True}}}}


def _skfile(length):
    return "x" * length


@pytest.mark.parametrize("length", [1, MAX_BASIS_LENGTH - 1, MAX_BASIS_LENGTH])
def test_skfile_within_limit_is_accepted(length):
    write_input(BASIS={"PTYPE": "BIO", "SKFILE": _skfile(length)}, **PARAMS)


@pytest.mark.parametrize("length", [MAX_BASIS_LENGTH + 1, 150])
def test_skfile_too_long_raises(length):
    with pytest.raises(ConfigError, match=rf"SKFILE is {length} characters long"):
        write_input(BASIS={"PTYPE": "BIO", "SKFILE": _skfile(length)}, **PARAMS)


def test_ptype_line_too_long_raises():
    # "PTYPE=" (6 chars) + 67 chars = 73 chars
    with pytest.raises(ConfigError, match="PTYPE"):
        write_input(BASIS={"PTYPE": "P" * 67, "SKFILE": "path"}, **PARAMS)


def test_list_basis_line_too_long_raises():
    basis = ["PTYPE=BIO\n" + _skfile(MAX_BASIS_LENGTH + 1)]
    with pytest.raises(ConfigError, match="BASIS BASIS"):
        write_input(BASIS=basis, **PARAMS)


def test_empty_basis_is_accepted():
    write_input(BASIS={"PTYPE": "", "SKFILE": ""}, **PARAMS)


def test_calculator_stops_at_construction(tmp_path):
    with pytest.raises(ConfigError):
        deMonNano(
            execut="deMonNano.x",
            workdir=str(tmp_path),
            basis={"PTYPE": "BIO", "SKFILE": _skfile(MAX_BASIS_LENGTH + 1)},
            **PARAMS,
        )
    assert not (tmp_path / "deMon.inp").exists()
