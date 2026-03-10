import os
import json

from deMonPy.modules.quench import _relax_geometry
from deMonPy.modules.ptmc import _ptmc
from deMonPy.modules.dyn import _dyn

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
                    
                }
            }
        }
    }
}



