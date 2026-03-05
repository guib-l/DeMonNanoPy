import os
import json

from deMonPy.modules.quench import _relax_geometry


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
}



