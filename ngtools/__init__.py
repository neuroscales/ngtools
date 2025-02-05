"""Miscelaneous tools to build and manipulate Neuroglancer scenes."""
__all__ = ["__version__"]

from ._version import __version__  # type: ignore

# Monkey patch neuroglancer

import threading
from neuroglancer.viewer_state import JsonObjectWrapper


def _is_compatible(obj, kls):
    if isinstance(obj, kls):
        return True
    bases = (base for base in kls.__bases__ if base is not object)
    return any(map(lambda x: _is_compatible(obj, x), bases))


def __patched_init__(self, json_data=None, _readonly=False, **kwargs):
    if json_data is None:
        json_data = {}
    elif _is_compatible(json_data, type(self)):  # !!! patched line
        json_data = json_data.to_json()
    elif not isinstance(json_data, dict):
        raise TypeError(type(json_data), type(self))
    object.__setattr__(self, "_json_data", json_data)
    object.__setattr__(self, "_cached_wrappers", dict())
    object.__setattr__(self, "_lock", threading.RLock())
    object.__setattr__(self, "_readonly", 1 if _readonly else False)
    for k in kwargs:
        setattr(self, k, kwargs[k])
    object.__setattr__(self, "_readonly", _readonly)


JsonObjectWrapper.__init__ = __patched_init__
