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

def map_to_list(f, l):
    """Does map operation and then converts map object to a list."""
    return list(map(f, l))

def map_to_ndarray(f, l):
    """Does map operation and then converts map object to a list."""
    return nd.array(map(f, l))

def location_to_ndarray(l):
    """Converts carla.Location to ndarray [x, y, z]"""
    return nd.array([l.x, l.y, l.z])

def transform_to_location_ndarray(t):
    """Converts carla.Transform to location ndarray [x, y, z]"""
    return location_to_ndarray(t.location)

def transforms_to_location_ndarray(ts):
    """Converts an iterable of carla.Transform to a ndarray of size (len(iterable), 3)"""
    return nd.array(map_to_list(transform_to_location_ndarray, ts))

def transform_to_origin(transform, origin):
    """Create an adjusted transformation relative to origin.
    Creates a new transformation (doesn't mutate the parameters).
    
    Parameters
    ----------
    transform : carla.Transform
        The transform we want to adjust
    origin : carla.Transform
        The origin we want to adjust the transform to

    Returns
    -------
    carla.Transform
        New transform with the origin as reference.
    """
    location = transform.location
    rotation = transform.rotation
    return carla.Transform(
            carla.Location(
                x=location.x - origin.location.x,
                y=location.y - origin.location.y,
                z=location.z - origin.location.z),
            carla.Rotation(
                pitch=rotation.pitch,
                yaw=rotation.yaw - origin.rotation.yaw,
                roll=rotation.roll))

