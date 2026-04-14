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


available_modules = {
    "opt":{
        "module":_relax_geometry,
        "args":{
            "DEMON_MODULE":{
                "ACTIVE":{
                    "OPT":{
                        "MAX":999,
                        "OUT":1,
                        "TRAJECTORY":True
                    },
                },
            }
        },
    },
    "ptmc":{
        "module":_ptmc,
        "args":{
            "DEMON_MODULE":{
                "ACTIVE":{
                    "PTMC":{
                        "MC":{
                            "MAX":30,
                            "SEED":True,
                            "WALL":8.
                        },
                        "MCTEMP":{
                            "TMC":300,
                            "NTEMP":10,
                            "GEOM":True,
                            "LINEAR":False,
                            "TEMPMIN":30,
                            "TEMPMAX":300,
                            "RESCALE":10,
                            "SDBG":False,
                            "OUT":1,
                            "SOUT":1
                        }
                    },
                },
            }
        }
    },
    "md":{
        "module":_dyn,
        "args":{
            "DEMON_MODULE":{
                "ACTIVE":{
                    "MD":{
                        "MDYNAMICS":{
                            "ZERO":True,
                            "RANDOM":300,
                            "RAN":False,
                            "READ":{
                                "VELOCITIES":[]
                            },
                            "RESET":False,
                            "WALL":None,
                            "EXP":None,
                            "ENER":None
                        },
                        "TIMESTEP":0.4,
                        "MDSTEP":{
                            "MAX":50,
                            "OUT":10,
                            "SOUT":1,
                            "TSIM":None
                        },
                        "MDTEMP":300   
                    } 
                }
            }
        }
    }
}



