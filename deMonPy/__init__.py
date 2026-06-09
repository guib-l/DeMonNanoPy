import os
import json

from deMonPy.modules.quench import _relax_geometry
from deMonPy.modules.ptmc import _ptmc
from deMonPy.modules.dyn import _dyn


"""
Available module in the deMonNanoAPI
 - opt  : Optimization
 - ptmc : Paralel Tempering Monte Carlo
 - md   : Simple molecular dynamics
"""
available_modules = {
    "OPT":None,
    "PTMC":None,
    "MD":None,
}

# Global configuration defaults
DEMON_EXECUTABLE = None
DEMON_BASIS = None


def configure(executable=None, basis=None):
    """Set global default values for executable and basis.

    Args:
        executable: Path to the deMonNano executable.
        basis: Basis configuration dictionary.
    """
    global DEMON_EXECUTABLE, DEMON_BASIS
    if executable is not None:
        DEMON_EXECUTABLE = executable
    if basis is not None:
        DEMON_BASIS = basis


def configure_from_file(path="global.json"):
    """Load global configuration from a JSON file.

    Args:
        path: Path to the JSON configuration file.
    """
    with open(path) as f:
        config = json.load(f)
    configure(
        executable=config.get("DEMON_EXECUTABLE"),
        basis=config.get("DEMON_BASIS")
    )

