"""DeMonError family raised by ``deMonPy.profile.Process.execute``.

Binary-free: the executable and working directory are plain files /
directories created in ``tmp_path``; no deMon calculation and no external
binary is ever run.  Each test trips one of the pre-flight checks so that
``subprocess.run`` is never reached.
"""

import re

import pytest

from deMonPy.exceptions import ConfigError, ExecuteFailed
from deMonPy.profile import Process


def test_missing_executable_raises_config_error(tmp_path):
    """A nonexistent executable path raises ConfigError carrying the path."""
    missing = tmp_path / "no-such-binary"
    process = Process(executable=missing, workdir=tmp_path)
    with pytest.raises(ConfigError, match=re.escape(str(missing))):
        process.execute()


def test_nonexecutable_executable_raises_execute_failed(tmp_path):
    """A file that exists but has no execute bit raises ExecuteFailed."""
    exe = tmp_path / "not-executable"
    exe.write_text("echo hi\n")
    exe.chmod(0o644)  # explicitly strip any execute bit
    process = Process(executable=exe, workdir=tmp_path)
    with pytest.raises(ExecuteFailed, match=re.escape(str(exe))):
        process.execute()


def test_directory_as_executable_raises_config_error(tmp_path):
    """A directory passed where an executable file is expected raises ConfigError."""
    directory = tmp_path / "not-a-file"
    directory.mkdir()
    process = Process(executable=directory, workdir=tmp_path)
    with pytest.raises(ConfigError, match=re.escape(str(directory))):
        process.execute()


def test_timeout_raises_execute_failed(tmp_path):
    """A process that exceeds its timeout raises ExecuteFailed, not TimeoutExpired.

    The fake executable sleeps far longer than the requested timeout so the
    test completes in ~1 s instead of waiting the full sleep.
    """
    exe = tmp_path / "sleep.exe"
    exe.write_text("#!/bin/sh\nsleep 30\n")
    exe.chmod(0o755)
    process = Process(executable=exe, workdir=tmp_path)
    with pytest.raises(ExecuteFailed, match=re.escape("timeout")) as excinfo:
        process.execute(timeout=1)
    message = str(excinfo.value)
    assert "timeout" in message
    assert "1" in message  # the requested timeout value
