"""Various utilities."""
# stdlib
import json
import logging
import socket
import sys
from urllib.parse import quote

# externals
import neuroglancer as ng

LOG = logging.getLogger(__name__)


NG_URLS = {
    "ng":
        "https://neuroglancer-demo.appspot.com/",
    "linc":
        "https://neuroglancer.lincbrain.org/cloudfront/frontend/index.html",
}


def neuroglancer_state_to_neuroglancer_url(
    state: ng.ViewerState,
    base_url: str = NG_URLS["ng"],
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
                # if there is a single argument and it has the right
                # type, just wrap it (no deep copy)
                if len(args) == 1 and isinstance(args[0], kls) and not kwargs:
                    wrapped = args[0]
                    if type(wrapped) is _NG_TYPE_WRAPPERS[kls]:
                        wrapped = wrapped._wrapped
                # otherwise, call the wraped class constructor (deep copy)
                else:
                    wrapped = kls(*args, **kwargs)
                object.__setattr__(self, "_wrapped", wrapped)

            def __fix_none_value__(self, name: str, value: object) -> object:
                # magic method for attributes that have a __default__
                if name.startswith("__"):
                    return value
                if value is None:
                    has_default = hasattr(self, f"__default_{name}__")
                    if has_default:
                        value = getattr(self, f"__default_{name}__")
                        setattr(self, name, value)
                        value = getattr(self._wrapped, name)
                return value

            def __getattribute__(self, name: str) -> object:
                # FIXME: I haven't managed to use __getattr__ only,
                # as it fails when the wrapped object implements its
                # own fancy __getattr__ (e.g. `ManagedLayer`).
                # It *might* be possible to solve this, but in the
                # meantime I had to overload __getattribute__. This
                # is annoying, as overloading __getattribute__ has
                # a non-zero performance cost.

                # magic objects -- never defer to _wrapped
                if name.startswith("__") or name == "_wrapped":
                    return object.__getattribute__(self, name)

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
                    # FIXME: I don't think we ever enter this branch
                    # -> should be dealt with in __getattribute__
                    value = self.__dict__[name]
                    return self.__fix_none_value__(name, value)
                if "_wrapped" in self.__dict__:
                    # FIXME: not entirely sure I need this test -- I
                    # should be able to assume _wrapped is available.
                    # And if not, we're happy with it raising an
                    # AttributeError anyway.
                    value = getattr(self._wrapped, name)
                    return self.__fix_none_value__(name, value)
                raise AttributeError(name)

            def __setattr__(self, name: str, value: object | type) -> None:
                if value is None:
                    value = self.__fix_none_value__(name, value)
                # check if a setter is available
                if hasattr(self, f"__set_{name}__"):
                    value = getattr(self, f"__set_{name}__")(value)
                # if attribute exists in the main object, set it there
                if name in self.__dict__:
                    # NOTE: I don't remember why I had to use the
                    # object method here.
                    return object.__setattr__(self, name, value)
                # if attribute exists in the warpped object, set it there
                if "_wrapped" in self.__dict__:
                    if hasattr(self._wrapped, name):
                        return setattr(self._wrapped, name, value)
                # fallback to default implementation.
                return super().__setattr__(name, value)

            def __delattr__(self, name: str) -> None:
                # if attribute exists in the main object, delete it there
                if name in self.__dict__:
                    return super().__delattr__(name)
                # if attribute exists in the wrapped object, delete it there
                if hasattr(self, "_wrapped") and hasattr(self._wrapped, name):
                    return delattr(self._wrapped, name)
                # fallback to default implementation.
                return super().__delattr__(name)

        _Wraps.__wrapper_class__ = _Wraps
        _Wraps.__wrapped_class__ = kls
        _NG_TYPE_WRAPPERS[kls] = _Wraps

    return _NG_TYPE_WRAPPERS[kls]


def find_available_port(port: int = 0, ip: str = "") -> tuple[int, str]:
    """Return an available port and the local IP."""
    try:
        s = socket.socket()
        s.bind((ip, port))
        ip = s.getsockname()[0]
        port = s.getsockname()[1]
        s.close()
    except OSError:
        port0 = port
        s = socket.socket()
        s.bind((ip, 0))
        ip = s.getsockname()[0]
        port = s.getsockname()[1]
        s.close()
        print(f'Port {port0} already in use. Use port {port} instead.',
              file=sys.stderr)
    return port, ip
