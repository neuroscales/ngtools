"""Various utilities."""
# stdlib
import json
from typing import Any
from urllib.parse import quote

# externals
import neuroglancer as ng

DEFAULT_URL = (
    "https://neuroglancer.lincbrain.org/cloudfront/frontend/index.html"
)


def neuroglancer_state_to_neuroglancer_url(
    state: ng.ViewerState,
    base_url: str = DEFAULT_URL,
) -> str:
    """
    Take the current state of the Neuroglancer viewer and convert it
    into a url-encoded value for the browser.
    """
    ordered_dict = state.to_json()

    # Convert the dictionary to a JSON string
    json_str = json.dumps(ordered_dict)

    # URL-encode the JSON string
    encoded_json = quote(json_str)

    # Construct and return the full URL
    return f"{base_url}#!{encoded_json}"


def Wraps(kls: type) -> type:
    """
    Return a wrapper type that inherits from `kls` but never calls
    `super().__init__()`.

    Instead, it stores an instance of the parent class and overloads
    attribute access so that attributes of the parent instance -- rather
    than `self` -- are accessed and set.

    This is necessary when wrapping neuroglancer types into our own types.
    """

    class _Wraps(kls):

        __name__ = "Wrapped" + kls.__name__
        __doc__ = kls.__doc__

        def __init__(self, *args, **kwargs) -> None:
            if args and isinstance(args[0], kls):
                self._wrapped = args[0]
            else:
                self._wrapped = kls(*args, **kwargs)

        def __getattribute__(self, name: str) -> object:
            try:
                return super().__getattribute__(f"__get_{name}")()
            except AttributeError:
                ...
            return super().__getattribute__(name)

        def __getattr__(self, name: str) -> Any:  # noqa: ANN401
            if name == "_wrapped":
                return super().__getattr__(name)
            if name in self.__dict__:
                return super().__getattr__(name)
            return getattr(self._wrapped, name)

        def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401
            if hasattr(self, f"__set_{name}"):
                value = getattr(self, f"__set_{name}")(value)
            if name == "_wrapped":
                return super().__setattr__(name, value)
            if name in self.__dict__:
                return super().__setattr__(name, value)
            if hasattr(self, "_wrapped") and hasattr(self._wrapped, name):
                return setattr(self._wrapped, name, value)
            return super().__setattr__(name, value)

        def __delattr__(self, name: str) -> None:
            if name in self.__dict__:
                return super().__delattr__(name)
            if hasattr(self, "_wrapped") and hasattr(self._wrapped, name):
                return delattr(self._wrapped, name)
            return super().__delattr__(name)

    return _Wraps
