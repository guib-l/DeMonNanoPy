
# Provisoir !!!
import json
import ase 
import numpy as np

class AseEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, ase.Atoms):
            return obj.__repr__()
        return super().default(obj)

# !!!





