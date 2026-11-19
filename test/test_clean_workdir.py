"""Safety guard of :meth:`~deMonPy.basic.Common.clean_workdir`.

Binary-free tests: the calculators are built without an executable
(``execut=None``) and only :meth:`clean_workdir` is called, so no deMon
binary is ever run.  ``clean_workdir`` must delete only the known
deMonNano artefacts from the working directory, and it must refuse --
raising :class:`~deMonPy.exceptions.ConfigError` **before any deletion**
-- when the resolved working directory is the filesystem root or the
user's home directory.

Safety of the root/home tests: the guard fires at the very top of
``clean_workdir`` (before the first ``glob``/``os.remove``), and the
calculators below are built with ``clean_repository=False`` so the
constructor never cleans either.  Nothing can be deleted, even if the
real home directory contains files matching the artefact patterns
(e.g. ``*.log``).
"""

from pathlib import Path

import pytest

from deMonPy.deMonNano import deMonNano
from deMonPy.exceptions import ConfigError


def _calc(workdir):
    """An executor-less :class:`~deMonPy.deMonNano.deMonNano` (never cleans at init)."""
    return deMonNano(
        execut=None,
        workdir=workdir,
        basis={"PTYPE": "BIO", "SKFILE": "slako"},
        DEMON_PARAMETERS={"ACTIVE": {"DFTB": {"SCC": True}}},
        DEMON_MODULE={"ACTIVE": {}},
        clean_repository=False,
    )


def test_clean_workdir_removes_artifacts_and_keeps_the_rest(tmp_path):
    """Artefacts are deleted; unrelated files and sub-directories survive."""
    calc = _calc(str(tmp_path))

    out = tmp_path / "deMon.out"
    log = tmp_path / "run.log"
    keep = tmp_path / "keep.txt"
    sub = tmp_path / "sub"
    out.write_text("done\n", encoding="utf-8")
    log.write_text("log line\n", encoding="utf-8")
    keep.write_text("keep me\n", encoding="utf-8")
    sub.mkdir()
    (sub / "inner.txt").write_text("nested\n", encoding="utf-8")

    calc.clean_workdir()

    assert not out.exists()
    assert not log.exists()
    assert keep.exists()
    assert (sub / "inner.txt").exists()


def test_clean_workdir_refuses_filesystem_root():
    """``workdir='/'`` raises ConfigError naming the resolved path.

    The guard raises before the glob loop (first statements of the
    method body), so nothing is deleted; the test therefore asserts the
    exception alone.
    """
    calc = _calc("/")

    with pytest.raises(ConfigError) as excinfo:
        calc.clean_workdir()

    message = str(excinfo.value)
    assert str(Path("/").expanduser().resolve()) in message
    assert "filesystem root" in message


def test_clean_workdir_refuses_home_directory():
    """``workdir=str(Path.home())`` raises ConfigError naming the resolved path.

    Safe by construction: the guard fires before the first ``glob`` and
    the calculator is built with ``clean_repository=False``, so no file
    of the real home can be touched -- even if it contains files
    matching the artefact patterns (e.g. ``*.log``).
    """
    home = str(Path.home())
    calc = _calc(home)

    with pytest.raises(ConfigError) as excinfo:
        calc.clean_workdir()

    message = str(excinfo.value)
    assert str(Path(home).expanduser().resolve()) in message
    assert "home directory" in message
