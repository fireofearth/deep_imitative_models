import json
import numpy as np
import os

class NumpyEncoder(json.JSONEncoder):
    """The encoding object used to serialize np.ndarrays"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_datum(datum, directory, filename):
    """Save datum

    Parameters
    ----------
    datum : dict
        The data to save
    directory : str
        The directory name to save the data.
    filename : str
        The filename to name the data
    """
    assert(os.path.isdir(directory))
    filename = filename if filename.endswith('.json') else f"{filename}.json"
    with open(os.path.join(directory, filename), 'w') as f:
            json.dump(datum, f, cls=NumpyEncoder)
