"""
Miscellaneous tools to build and manipulate Neuroglancer scenes.

Modules
-------
scene
    Extension to `ng.ViewerState` that adds convenience methods to build
    and manipulate Neuroglancer scenes.
layers
    Extensions to `ng.Layer` that simplify the creation of common layer types.
datasources
    Extensions to `ng.LayerDataSource` that exposes data source properties
    usually not available through Neuroglancer's Python API.
local
    Utilities to create and interact with local Neuroglancer intances
    and manipulate local files.
transforms
    Utilities to create and manipulate `ng.CoordinateTransform` objects.
spaces
    Definition of standard coordinate spaces, such as RAS, in the form of
    `ng.CoordinateSpace`.
units
    Conversion between units used in different formats.
shaders
    Definition of custom GLSL shaders for Neuroglancer layers.
cmdata
    Matplotlib colormaps that can be used in Neuroglancer shaders.
opener
    Flexible file opener that supports local and remote files.
dandisf
    A `fsspec` filesystem for [DANDI](https://dandiarchive.org).
protocols
    Definition of ngtools-specific URI protocols.
utils
    Miscellaneous utility functions.

"""
# ruff: noqa
# flake8: noqa
__all__ = ["__version__"]

# import to trigger fsspec registration
from . import dandifs  # noqa: F401

# version
from ._version import __version__  # type: ignore

# monkey patch neuroglancer
from neuroglancer import StackLayout as _StackLayout
from functools import wraps

_old_stack_to_json = _StackLayout.to_json


@wraps(_old_stack_to_json)
def _stack_to_json(self: _StackLayout) -> dict:
    json = _old_stack_to_json(self)
    if "children" in json:
        json["children"] = [
            {"type": child} if isinstance(child, str) else child
            for child in json["children"]
        ]
    return json


_StackLayout.to_json = _stack_to_json
