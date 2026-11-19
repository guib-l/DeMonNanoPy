"""Unit tests for the :mod:`demon_env` helper (no deMon binary required).

The real ``deMonPy.configure_from_file`` is never called here: it is always
monkeypatched with a recorder, so these tests run in any environment.
"""

import json

import pytest

import deMonPy
from demon_env import configure_or_skip


def _write_config(directory, payload):
    config = directory / "global.json"
    config.write_text(json.dumps(payload), encoding="utf-8")
    return config


def test_missing_config_file_skips(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(pytest.skip.Exception) as excinfo:
        configure_or_skip("global.json")
    message = str(excinfo.value)
    assert str(tmp_path) in message
    assert "INSTALL.md" in message


def test_missing_basis_key_skips(tmp_path, monkeypatch):
    _write_config(tmp_path, {"DEMON_EXECUTABLE": "/bin/true"})
    monkeypatch.chdir(tmp_path)
    with pytest.raises(pytest.skip.Exception) as excinfo:
        configure_or_skip("global.json")
    assert "DEMON_BASIS" in str(excinfo.value)


def test_nonexistent_executable_skips(tmp_path, monkeypatch):
    _write_config(tmp_path, {"DEMON_EXECUTABLE": "/nonexistent/deMon.x",
                             "DEMON_BASIS": str(tmp_path)})
    monkeypatch.chdir(tmp_path)
    with pytest.raises(pytest.skip.Exception) as excinfo:
        configure_or_skip("global.json")
    assert "/nonexistent/deMon.x" in str(excinfo.value)


def test_missing_basis_directory_skips(tmp_path, monkeypatch):
    basis = tmp_path / "no_such_dir"
    _write_config(tmp_path, {"DEMON_EXECUTABLE": "/bin/true", "DEMON_BASIS": str(basis)})
    monkeypatch.chdir(tmp_path)
    with pytest.raises(pytest.skip.Exception) as excinfo:
        configure_or_skip("global.json")
    assert str(basis) in str(excinfo.value)


def test_valid_config_calls_configure_from_file(tmp_path, monkeypatch):
    _write_config(tmp_path, {"DEMON_EXECUTABLE": "/bin/true", "DEMON_BASIS": str(tmp_path)})
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(deMonPy, "configure_from_file", lambda path: calls.append(path))
    configure_or_skip("global.json")
    assert calls == ["global.json"]
