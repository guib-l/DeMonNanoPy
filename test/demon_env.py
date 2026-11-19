"""Helper to configure the deMon environment in binary-dependent tests.

Purpose:
    Test modules that depend on the deMonNano executable and the basis
    files load the project configuration (``global.json``) at import time.
    When the deMon environment is not configured, such modules must not
    error at import: they call :func:`configure_or_skip` at module level,
    which either configures ``deMonPy`` exactly as before, or skips the
    whole module (``pytest.skip`` with ``allow_module_level=True``) with a
    clear message explaining what is missing.

Usage:
    In each binary-dependent test module, at the top of the file::

        from demon_env import configure_or_skip

        configure_or_skip("global.json")
"""

import json
import os

import pytest

import deMonPy

_REQUIRED_KEYS = ("DEMON_EXECUTABLE", "DEMON_BASIS")


def configure_or_skip(path: str = "global.json") -> None:
    """Configure ``deMonPy`` from the JSON file at *path*, or skip the module.

    The checks, in order:

    1. *path* exists (resolved against the current working directory);
    2. it is valid JSON (genuine JSON errors propagate);
    3. it is a JSON object with string values for ``DEMON_EXECUTABLE``
       and ``DEMON_BASIS``;
    4. ``DEMON_EXECUTABLE`` (after ``os.path.expanduser``) is an existing,
       executable file;
    5. ``DEMON_BASIS`` (after ``os.path.expanduser``) is an existing
       directory.

    When a check fails, the module is skipped via
    ``pytest.skip(..., allow_module_level=True)`` with a message naming the
    problem and pointing to ``INSTALL.md``. When all checks pass,
    ``deMonPy.configure_from_file(path)`` is called (previous behavior)
    and this function returns.

    Args:
        path: Path to the JSON configuration file, CWD-relative.
    """
    if not os.path.exists(path):
        pytest.skip(
            "deMon config file not found: looked for "
            f"{path!r} in current working directory {os.getcwd()!r}. "
            "See INSTALL.md to set up the deMon environment.",
            allow_module_level=True,
        )
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        pytest.skip(
            f"deMon config file {path!r} must be a JSON object with string "
            "values for 'DEMON_EXECUTABLE' and 'DEMON_BASIS', but it is a "
            f"{type(config).__name__}. See INSTALL.md.",
            allow_module_level=True,
        )
    missing = [key for key in _REQUIRED_KEYS if not isinstance(config.get(key), str)]
    if missing:
        pytest.skip(
            f"deMon config file {path!r} is missing string value(s): "
            f"{', '.join(missing)}. See INSTALL.md to set up the deMon "
            "environment.",
            allow_module_level=True,
        )
    executable = os.path.expanduser(config["DEMON_EXECUTABLE"])
    if not (os.path.isfile(executable) and os.access(executable, os.X_OK)):
        pytest.skip(
            "deMon executable missing or not executable: "
            f"{executable!r} (declared in {path!r}). "
            "See INSTALL.md to set up the deMon environment.",
            allow_module_level=True,
        )
    basis = os.path.expanduser(config["DEMON_BASIS"])
    if not os.path.isdir(basis):
        pytest.skip(
            f"deMon basis directory not found: {basis!r} (declared in "
            f"{path!r}). See INSTALL.md to set up the deMon environment.",
            allow_module_level=True,
        )
    deMonPy.configure_from_file(path)
