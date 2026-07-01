#!/usr/bin/env python3
import os
import subprocess
from functools import wraps
from pathlib import Path

from deMonPy.exceptions import ExecuteFailed


def read_json(filename=""):
    import json

    data = {}
    try:
        with open(filename, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"File {filename} not found.")
    except json.JSONDecodeError:
        print("Invalid JSON format.")
    return data


class Process:
    def __init__(self, executable, workdir="", omp_threads=1, prefix="DEMON"):

        self.executable = Path(executable).expanduser() if executable else None
        self.workdir = Path(workdir).expanduser() if workdir else Path(".")
        self.prefix = prefix
        self.omp_threads = omp_threads

    def execute(self, check=True, timeout=None):

        if self.executable is None:
            raise EnvironmentError("No executable configured")

        exe = self.executable.resolve()
        if not exe.is_file():
            raise FileNotFoundError(f"Executable not found: {exe}")
        if not os.access(exe, os.X_OK):
            raise PermissionError(f"Executable not runnable: {exe}")

        wd = self.workdir.resolve()
        if not wd.is_dir():
            raise NotADirectoryError(f"Working directory missing: {wd}")

        env = {**os.environ, "OMP_NUM_THREADS": str(self.omp_threads)}

        try:
            result = subprocess.run(
                [str(exe)],
                cwd=str(wd),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                shell=False,
            )
        except OSError as err:
            raise EnvironmentError(f'Failed to execute "{exe}"') from err

        if check and result.returncode != 0:
            raise ExecuteFailed(
                f'Calculator "{self.prefix}" failed with return code '
                f"{result.returncode} in {wd}\nstderr:\n{result.stderr}"
            )

        return result


def assert_flags(myset):
    def decorateur(func):

        @wraps(func)
        def wrapper(self, *args, **kwargs):

            _flags = getattr(self, "flags")

            if isinstance(myset, list):
                for ms in myset:
                    if ms not in _flags:
                        return
            else:
                if myset not in _flags:
                    return

            return func(self, *args, **kwargs)

        return wrapper

    return decorateur


def exclude_flags(myset):
    def decorateur(func):

        @wraps(func)
        def wrapper(self, *args, **kwargs):

            _flags = getattr(self, "flags")

            if isinstance(myset, list):
                for ms in myset:
                    if ms in _flags:
                        return
            else:
                if myset in _flags:
                    return

            return func(self, *args, **kwargs)

        return wrapper

    return decorateur
