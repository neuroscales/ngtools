"""Utility to deal with optional dependencies."""
# stdlib
import importlib
from types import ModuleType


def _try_import(
    path: str,
    keys: str | list[str] | None = None,
    _as: bool = False
) -> ModuleType | object | None | tuple[ModuleType | object | None, ...]:
    """Try to import from a module.

    Example
    -------
    ```
    # import os.path
    os = try_import('os.path')

    # from os import path
    path = try_import('os', 'path')
    ```

    Parameters
    ----------
    path : str
        Path to module or variable in a module
    keys : str or list[str], optional
        Keys to load from the module
    _as : bool, defualt=True
        If False, perform recursive assignement as in
        >> # equivalent to: `import pack.sub.mod`
        >> pack = try_import('pack.sub.mod', _as=False)
        Else, it will look like a renamed import:
        >> # equivalent to: `import pack.sub.mod as my_mod`
        >> my_mod = try_import('pack.sub.mod', _as=True)


    Returns
    -------
    loaded_stuff : module or object or tuple
        A tuple is returned if `keys` is a list.
        Return None if import fails.

    """
    if path is None:
        return None

    def fail(keys: str | list[str] | None) -> None | list[None]:
        if keys is None or isinstance(keys, str):
            return None
        else:
            keys = list(keys)
            return [None]*len(keys)

    def try_import_module(path: str) -> ModuleType | None:
        try:
            return importlib.import_module(path)
        except (ImportError, ModuleNotFoundError):
            return None

    # check if the base package exists
    pack = path.split('.')[0]
    try:
        __import__(pack)
    except (ImportError, ModuleNotFoundError):
        return fail(keys)

    if _as:
        # import a module
        module = try_import_module(path)
        if not module:
            return fail(keys)
        # optional: extract attributes
        if keys is not None:
            if isinstance(keys, str):
                return getattr(module, keys)
            else:
                return tuple(getattr(module, key) for key in keys)
        return module
    else:
        # recursive import
        path = path.split('.')
        mod0 = try_import_module(path[0])
        if not mod0:
            return fail(keys)
        cursor = mod0
        for i in range(1, len(path)):
            mod1 = try_import_module('.'.join(path[:i+1]))
            if not mod1:
                return fail(keys)
            setattr(cursor, path[i], mod1)
            cursor = getattr(cursor, path[i])
        return mod0


def try_import(path: str, fallback: str | None = None) -> ModuleType | None:
    """Try to import a module.

    Example
    -------
    ```
    # import os
    os = try_import('os')

    # import os.path
    os = try_import('os.path')
    ```

    Parameters
    ----------
    path : str
        Path to module.
    fallaback: str, optional
        Path to fallback module

    Returns
    -------
    loaded_module : module | None
        Return `None` if import fails.

    """
    return _try_import(path) or _try_import(fallback)


def try_import_as(path: str, fallback: str | None = None) -> ModuleType | None:
    """Try to import a module and alias it.

    Example
    -------
    ```
    # import os.path as op
    op = try_import_as('os.path')
    ```

    Parameters
    ----------
    path : str
        Path to module.
    fallaback: str, optional
        Path to fallback module

    Returns
    -------
    loaded_module : module | None
        Return `None` if import fails.

    """
    return _try_import(path, _as=True) or _try_import(fallback, _as=True)


def try_from_import(
    path: str, keys: str | list[str], fallback: str | None = None,
) -> ModuleType | object | None | tuple[ModuleType | object | None, ...]:
    """Try to import attributes from a module.

    Example
    -------
    ```
    # from os import path
    path = try_from_import('os', 'path')

    # from os.path import join, splitext
    join, splitext = try_from_import('os.path', ['join', 'splitext'])

    # from os.path import join as J, splitext as S
    J, S = try_from_import('os.path', ['join', 'splitext'])
    ```

    Parameters
    ----------
    path : str
        Path to module.
    keys : str | list[str]
        Attributes to import from module
    fallaback: str, optional
        Path to fallback module

    Returns
    -------
    loaded_module : module | object | None | list[module | object | None]
        Return `None` if import fails.
        Return a list of objects if `keys` is a list.

    """
    return (
        _try_import(path, keys, _as=True) or
        _try_import(fallback, keys, _as=True)
    )
