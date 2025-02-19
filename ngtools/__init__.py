"""Miscelaneous tools to build and manipulate Neuroglancer scenes."""
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
