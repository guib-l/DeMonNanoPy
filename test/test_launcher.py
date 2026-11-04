"""Launcher support (mpirun, srun...) in deMonPy.profile.Process.

These tests need neither deMonNano nor MPI: a fake `mpirun` shell script,
put first in PATH, records its arguments and OMP_NUM_THREADS then runs
the executable given as last argument.
"""

import os
import stat

import pytest

import deMonPy
from deMonPy.deMonNano import deMonNano
from deMonPy.exceptions import ConfigError, ExecuteFailed
from deMonPy.profile import Process

PARAMS = {"DEMON_PARAMETERS": {"ACTIVE": {"DFTB": {"SCC": True}}}}


def _script(path, body):
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.fixture
def fake_mpi(tmp_path, monkeypatch):
    """Fake `mpirun` recording argv + OMP_NUM_THREADS, then running the exe."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _script(
        bindir / "mpirun",
        'echo "$@" > launcher.log\n'
        'echo "${OMP_NUM_THREADS:-unset}" >> launcher.log\n'
        'for last; do :; done\n'
        'exec "$last"\n',
    )
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    exe = _script(tmp_path / "deMon.x", "touch ran.flag\n")
    return exe, tmp_path


def _read_log(wd):
    return (wd / "launcher.log").read_text().splitlines()


# ----------------------------------------------------------------- building
def test_no_launcher_keeps_previous_behaviour():
    p = Process("deMon.x", omp_threads=4)
    assert p.build_command("/x/deMon.x") == ["/x/deMon.x"]
    assert p.build_env()["OMP_NUM_THREADS"] == "4"


def test_placeholders_and_exe_appended(fake_mpi):
    p = Process(
        "deMon.x",
        omp_threads=4,
        nprocs=2,
        launcher="mpirun -np {nprocs} --map-by slot:PE={omp_threads}",
    )
    assert p.build_command("/x/deMon.x") == [
        "mpirun", "-np", "2", "--map-by", "slot:PE=4", "/x/deMon.x",
    ]


def test_explicit_exe_placeholder(fake_mpi):
    p = Process("deMon.x", launcher=["mpirun", "{exe}", "--extra"])
    assert p.build_command("/x/deMon.x") == ["mpirun", "/x/deMon.x", "--extra"]


def test_omp_threads_exported_with_launcher(fake_mpi):
    p = Process("deMon.x", omp_threads=3, launcher="mpirun -np 1")
    assert p.build_env()["OMP_NUM_THREADS"] == "3"


def test_omp_threads_none_not_exported(fake_mpi, monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    p = Process("deMon.x", omp_threads=None, launcher="mpirun -np 1")
    assert "OMP_NUM_THREADS" not in p.build_env()


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"launcher": "does-not-exist-xyz"}, "not found"),
        ({"launcher": 42}, "string or a list"),
        ({"launcher": ["mpirun", 1]}, "string or a list"),
        ({"nprocs": 2}, "requires a launcher"),
        ({"nprocs": 2, "launcher": "env"}, "requires a launcher"),
        ({"nprocs": 0}, "positive integer"),
        ({"omp_threads": 0}, "positive integer"),
        ({"omp_threads": None, "launcher": "env N={omp_threads}"}, "omp_threads is None"),
    ],
)
def test_invalid_configuration_raises(kwargs, match):
    with pytest.raises(ConfigError, match=match):
        Process("deMon.x", **kwargs)


def test_empty_launcher_means_no_launcher():
    assert Process("deMon.x", launcher="").launcher is None
    assert Process("deMon.x", launcher=[]).launcher is None


# ---------------------------------------------------------------- execution
def test_execute_through_mpirun(fake_mpi):
    exe, wd = fake_mpi
    Process(
        exe, workdir=wd, omp_threads=3, nprocs=2,
        launcher="mpirun -np {nprocs} --bind-to core",
    ).execute()
    argv, omp = _read_log(wd)
    assert argv == f"-np 2 --bind-to core {exe.resolve()}"
    assert omp == "3"
    assert (wd / "ran.flag").exists()


def test_execute_failure_reports_command(fake_mpi):
    _, wd = fake_mpi
    bad = _script(wd / "bad.x", "exit 3\n")
    with pytest.raises(ExecuteFailed, match=r"command: mpirun -np 1 .*bad\.x"):
        Process(bad, workdir=wd, launcher="mpirun -np {nprocs}").execute()


# --------------------------------------------------------------- plumbing
def test_calculator_reads_launcher_parameters(fake_mpi):
    exe, wd = fake_mpi
    calc = deMonNano(
        execut=str(exe),
        workdir=str(wd),
        omp_threads=2,
        basis={"PTYPE": "", "SKFILE": ""},
        DEMON_LAUNCHER="mpirun -np {nprocs}",
        DEMON_NPROCS=4,
        **PARAMS,
    )
    assert calc.process.build_command(exe) == ["mpirun", "-np", "4", str(exe)]
    assert "DEMON_LAUNCHER" not in calc.parameters
    assert "DEMON_NPROCS" not in calc.parameters


def test_calculator_uses_global_defaults(fake_mpi, monkeypatch):
    exe, wd = fake_mpi
    monkeypatch.setattr(deMonPy, "DEMON_LAUNCHER", "mpirun -np {nprocs}")
    monkeypatch.setattr(deMonPy, "DEMON_NPROCS", 2)
    calc = deMonNano(
        execut=str(exe), workdir=str(wd),
        basis={"PTYPE": "", "SKFILE": ""}, **PARAMS,
    )
    assert calc.process.build_command(exe) == ["mpirun", "-np", "2", str(exe)]


def test_invalid_launcher_stops_calculator_construction(tmp_path):
    with pytest.raises(ConfigError):
        deMonNano(
            execut="deMon.x", workdir=str(tmp_path),
            basis={"PTYPE": "", "SKFILE": ""},
            DEMON_LAUNCHER="does-not-exist-xyz", **PARAMS,
        )


def test_configure_from_file(tmp_path, monkeypatch, fake_mpi):
    for name, value in [("DEMON_EXECUTABLE", None), ("DEMON_BASIS", None),
                        ("DEMON_LAUNCHER", None), ("DEMON_NPROCS", 1)]:
        monkeypatch.setattr(deMonPy, name, value)
    cfg = tmp_path / "global.json"
    cfg.write_text(
        '{"DEMON_EXECUTABLE": "deMon.x", "DEMON_BASIS": "basis",'
        ' "DEMON_LAUNCHER": "mpirun -np {nprocs}", "DEMON_NPROCS": 3}'
    )
    deMonPy.configure_from_file(str(cfg))
    assert deMonPy.DEMON_LAUNCHER == "mpirun -np {nprocs}"
    assert deMonPy.DEMON_NPROCS == 3
