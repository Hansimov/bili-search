import numpy as np
import Pyro5.api


def ndarray_to_dict(arr):
    """convert numpy.ndarray to dcit"""
    return {
        "__class__": "numpy.ndarray",  # must include this to notify Pyro to call dict_to_ndarray
        "__ndarray__": arr.tolist(),
        "dtype": str(arr.dtype),
        "shape": arr.shape,
    }


def dict_to_ndarray(classname, d):
    """reconstruct numpy.ndarray from dict"""
    if "__ndarray__" in d:
        return np.array(d["__ndarray__"], dtype=d["dtype"]).reshape(d["shape"])
    else:
        return d


def register_pyro_apis():
    Pyro5.api.register_class_to_dict(np.ndarray, ndarray_to_dict)
    Pyro5.api.register_dict_to_class("numpy.ndarray", dict_to_ndarray)
