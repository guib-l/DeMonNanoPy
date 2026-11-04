#!/usr/bin/env python3
import os
import shlex
import shutil
import subprocess
from functools import wraps
from pathlib import Path

from deMonPy.exceptions import ConfigError, ExecuteFailed


class Process:
    """Run the deMon/deMonNano binary, directly or through a launcher.

    The launcher is a command prefix (``mpirun``, ``srun``, ``numactl``...).
    The following placeholders are substituted in each of its tokens:

    * ``{nprocs}``      -- number of processes (:attr:`nprocs`);
    * ``{omp_threads}`` -- OpenMP threads per process (:attr:`omp_threads`);
    * ``{exe}``         -- executable path; appended at the end when absent.

    ``OMP_NUM_THREADS`` is exported whenever :attr:`omp_threads` is not
    ``None``; the launcher only handles process placement / CPU binding.
    With ``omp_threads=None`` the variable is left untouched and the OpenMP
    runtime follows the affinity mask given by the launcher.

    Examples::

        # Open MPI: omp_threads cores reserved for each process
        launcher="mpirun -np {nprocs} --map-by slot:PE={omp_threads} --bind-to core"
        # MPICH / Intel MPI (Hydra)
        launcher="mpirun -np {nprocs} -bind-to core:{omp_threads}"
        # Slurm
        launcher="srun -n {nprocs} -c {omp_threads} --cpu-bind=cores"
    """

    def __init__(
        self,
        executable,
        workdir="",
        omp_threads=1,
        prefix="DEMON",
        launcher=None,
        nprocs=1,
    ):
        self.executable = Path(executable).expanduser() if executable else None
        self.workdir = Path(workdir).expanduser() if workdir else Path(".")
        self.prefix = prefix
        self.omp_threads = (
            None if omp_threads is None else self._positive_int("omp_threads", omp_threads)
        )
        self.nprocs = self._positive_int("nprocs", nprocs)
        self.launcher = self._parse_launcher(launcher)
        self._check_consistency()

    # ------------------------------------------------------------------ checks
    @staticmethod
    def _positive_int(name, value):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigError(f"{name} must be a positive integer, got {value!r}")
        return value

    @staticmethod
    def _parse_launcher(launcher):
        """Normalise the launcher into a list of tokens, or ``None``."""
        if launcher is None:
            return None
        if isinstance(launcher, str):
            tokens = shlex.split(launcher)
        elif isinstance(launcher, (list, tuple)) and all(isinstance(t, str) for t in launcher):
            tokens = list(launcher)
        else:
            raise ConfigError(
                f"DEMON_LAUNCHER must be a string or a list of strings, got {launcher!r}"
            )
        if not tokens:
            return None
        if shutil.which(tokens[0]) is None:
            raise ConfigError(f"Launcher command not found in PATH: '{tokens[0]}'")
        return tokens

    def _uses(self, placeholder):
        return self.launcher is not None and any(placeholder in t for t in self.launcher)

    def _check_consistency(self):
        if self.nprocs > 1 and not self._uses("{nprocs}"):
            raise ConfigError(
                f"nprocs={self.nprocs} requires a launcher using '{{nprocs}}' "
                "(e.g. 'mpirun -np {nprocs}')"
            )
        if self.omp_threads is None and self._uses("{omp_threads}"):
            raise ConfigError("Launcher uses '{omp_threads}' but omp_threads is None")

    # ---------------------------------------------------------------- command
    def build_command(self, exe):
        """Return the argv list used to start *exe*."""
        exe = str(exe)
        if self.launcher is None:
            return [exe]

        values = {
            "{exe}": exe,
            "{nprocs}": str(self.nprocs),
            "{omp_threads}": str(self.omp_threads),
        }

        def fill(token):
            for key, val in values.items():
                token = token.replace(key, val)
            return token

        command = [fill(t) for t in self.launcher]
        if not self._uses("{exe}"):
            command.append(exe)
        return command

    def build_env(self):
        """Return the environment of the child process."""
        env = dict(os.environ)
        if self.omp_threads is not None:
            env["OMP_NUM_THREADS"] = str(self.omp_threads)
        return env

    # ---------------------------------------------------------------- execute
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

        command = self.build_command(exe)

        try:
            result = subprocess.run(
                command,
                cwd=str(wd),
                env=self.build_env(),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                shell=False,
            )
        except OSError as err:
            raise EnvironmentError(f'Failed to execute "{shlex.join(command)}"') from err

        if check and result.returncode != 0:
            raise ExecuteFailed(
                f'Calculator "{self.prefix}" failed with return code '
                f"{result.returncode} in {wd}\n"
                f"command: {shlex.join(command)}\nstderr:\n{result.stderr}"
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
