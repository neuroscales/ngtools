"""Miscelaneous tools to build and manipulate Neuroglancer scenes."""
__all__ = ["__version__"]

# import to trigger fsspec registration
from . import dandifs  # noqa: F401

# version
from ._version import __version__  # type: ignore
