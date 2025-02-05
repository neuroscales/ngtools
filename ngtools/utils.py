"""Various utilities."""
# stdlib
import functools
import json
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


def Wraps(kls):

    class _Wraps(kls):

        __name__ = "Wrapped" + kls.__name__

        def __init__(self, *args, **kwargs):
            if args and isinstance(args[0], kls):
                self._wrapped = args[0]
            else:
                self._wrapped = kls(*args, **kwargs)

        def __getattr__(self, name):
            if name == "_wrapped":
                return super().__getattr__(name)
            if name in self.__dict__:
                return super().__getattr__(name)
            return getattr(self._wrapped, name)

        def __setattr__(self, name, value):
            if name == "_wrapped":
                return super().__setattr__(name, value)
            if name in self.__dict__:
                return super().__setattr__(name, value)
            if hasattr(self, "_wrapped") and hasattr(self._wrapped, name):
                return setattr(self._wrapped, name, value)
            return super().__setattr__(name, value)

        def __delattr__(self, name):
            if name in self.__dict__:
                return super().__delattr__(name)
            if hasattr(self, "_wrapped") and hasattr(self._wrapped, name):
                return delattr(self._wrapped, name)
            return super().__delattr__(name)

    return _Wraps
