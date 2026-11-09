import json
import os

__version__ = "0.1.1"

from deMonPy.modules.dyn import _dyn
from deMonPy.modules.ptmc import _ptmc
from deMonPy.modules.quench import _relax_geometry

"""
Available module in the deMonNanoAPI
 - opt  : Optimization
 - ptmc : Parallel Tempering Monte Carlo
 - md   : Simple molecular dynamics
"""
available_modules = {
    "opt": {"module": _relax_geometry, "args": {}},
    "ptmc": {"module": _ptmc, "args": {}},
    "md": {"module": _dyn, "args": {}},
}

# Global configuration defaults
DEMON_EXECUTABLE = None
DEMON_BASIS = None
DEMON_BASIS_EXT = None
DEMON_LAUNCHER = None
DEMON_NPROCS = 1


def configure(executable=None, basis=None, launcher=None, nprocs=None):
    """Set global default values for executable, basis and launcher.

    Args:
        executable: Path to the deMonNano executable.
        basis: Basis configuration dictionary.
        launcher: Command prefix used to start the executable
            (e.g. ``"mpirun -np {nprocs}"``), see
            :class:`deMonPy.profile.Process`.
        nprocs: Number of processes substituted for ``{nprocs}``.
    """
    global DEMON_EXECUTABLE, DEMON_BASIS, DEMON_LAUNCHER, DEMON_NPROCS
    if executable is not None:
        DEMON_EXECUTABLE = executable
    if basis is not None:
        DEMON_BASIS = basis
    if launcher is not None:
        DEMON_LAUNCHER = launcher
    if nprocs is not None:
        DEMON_NPROCS = nprocs


def configure_from_file(path="global.json"):
    """Load global configuration from a JSON file.

    Args:
        path: Path to the JSON configuration file.
    """
    if not os.path.exists(path):
        raise FileExistsError(f"File globals.json didn't exists at {os.path.abspath(path)}")
    with open(path) as f:
        config = json.load(f)

    basis_str = os.path.abspath(config.get("DEMON_BASIS"))
    # if len(basis_str) > 40:
    #    basis_str = os.path.relpath(basis_str)

    configure(
        executable=os.path.abspath(config.get("DEMON_EXECUTABLE")),
        basis=basis_str,
        launcher=config.get("DEMON_LAUNCHER"),
        nprocs=config.get("DEMON_NPROCS"),
    )
