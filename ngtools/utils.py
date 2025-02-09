"""Various utilities."""
# stdlib
import json
import logging
from urllib.parse import quote

# externals
import neuroglancer as ng

LOG = logging.getLogger(__name__)

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


_NG_TYPE_WRAPPERS = {}


def Wraps(kls: type) -> type:
    """
    Return a wrapper type that inherits from `kls` but never calls
    `super().__init__()`.

    Instead, it stores an instance of the parent class and overloads
    attribute access so that attributes of the parent instance -- rather
    than `self` -- are accessed and set.

    This is necessary when wrapping neuroglancer types into our own types.
    """
    if kls not in _NG_TYPE_WRAPPERS:

        class _Wraps(kls):

            __name__ = "Wrapped" + kls.__name__
            __doc__ = kls.__doc__

            def __init__(self, *args, **kwargs) -> None:
                if args and isinstance(args[0], kls):
                    wrapped = args[0]
                    if type(wrapped) is _NG_TYPE_WRAPPERS[kls]:
                        wrapped = wrapped._wrapped
                else:
                    wrapped = kls(*args, **kwargs)
                object.__setattr__(self, "_wrapped", wrapped)

            def __fix_none_value__(self, name: str, value: object) -> object:
                if name.startswith("__"):
                    return value
                if value is None:
                    has_default = hasattr(self, f"__default_{name}__")
                    if has_default:
                        value = getattr(self, f"__default_{name}__")
                        setattr(self, name, value)
                        value = getattr(self._wrapper, name)
                return value

            def __getattribute__(self, name: str) -> object:
                # magic objects -- never defer to _wrapped
                if name.startswith("__"):
                    return super().__getattribute__(name)
                # check if a getter is available
                try:
                    value = object.__getattribute__(self, f"__get_{name}__")()
                    return self.__fix_none_value__(name, value)
                except AttributeError:
                    ...
                # check if attribute exists in _wrapped
                if name != "_wrapped" and "_wrapped" in self.__dict__:
                    try:
                        value = self._wrapped.__getattribute__(name)
                        return self.__fix_none_value__(name, value)
                    except AttributeError:
                        ...
                # fallback to default implementation
                value = object.__getattribute__(self, name)
                return self.__fix_none_value__(name, value)

            def __getattr__(self, name: str) -> object | type:
                if name in self.__dict__:
                    return self.__fix_none_value__(name, self.__dict__[name])
                value = getattr(self._wrapped, name)
                return self.__fix_none_value__(name, value)

            def __setattr__(self, name: str, value: object | type) -> None:
                if hasattr(self, f"__set_{name}"):
                    value = getattr(self, f"__set_{name}__")(value)
                if name in self.__dict__:
                    return object.__setattr__(self, name, value)
                if hasattr(self._wrapped, name):
                    return setattr(self._wrapped, name, value)
                return super().__setattr__(name, value)

            def __delattr__(self, name: str) -> None:
                if name in self.__dict__:
                    return super().__delattr__(name)
                if hasattr(self, "_wrapped") and hasattr(self._wrapped, name):
                    return delattr(self._wrapped, name)
                return super().__delattr__(name)

        _Wraps.__wrapper_class__ = _Wraps
        _Wraps.__wrapped_class__ = kls
        _NG_TYPE_WRAPPERS[kls] = _Wraps

    return _NG_TYPE_WRAPPERS[kls]
