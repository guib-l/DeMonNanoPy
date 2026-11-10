"""Behaviour of the ``ignore_parse_errors`` keyword of the output parsers."""

import pytest

from deMonPy.demon.output import read_output as read_output_demon
from deMonPy.demonnano.output import read_output
from deMonPy.exceptions import OutputParseError
from deMonPy.ioread import safe_parse


class BrokenParser(read_output):
    @safe_parse("frequency", "zpe")
    def read_freq(self):
        self.complet_results["frequency"] = [1.0]
        raise ValueError("cannot convert")


def test_default_ignores_and_sets_none(tmp_path):
    wo = BrokenParser(workdir=str(tmp_path))
    assert wo.ignore_parse_errors is True
    assert wo.read_freq() is None
    # partially filled keys are reset to None
    assert wo.complet_results["frequency"] is None
    assert wo.complet_results["zpe"] is None
    (err,) = wo.complet_results["errors"]
    assert err["kind"] == "parse"
    assert "read_freq" in err["message"]


def test_raise_when_disabled(tmp_path):
    wo = BrokenParser(workdir=str(tmp_path), ignore_parse_errors=False)
    with pytest.raises(ValueError, match="cannot convert"):
        wo.read_freq()


@pytest.mark.parametrize("parser", [read_output, read_output_demon])
def test_missing_output_always_raises(tmp_path, parser):
    wo = parser(workdir=str(tmp_path), ignore_parse_errors=True)
    with pytest.raises(OutputParseError):
        wo.read_file()
