"""
Utilities to manipulate spaces (`ng.CoordinateSpace`) in neuroglancer.

Functions
---------
convert_space
    Convert units of a coordinate space.
normalize_space
    Ensure that dimensions have SI units (without prefix).
name_compact2full
    Convert a compact axis name (`"RAS"`) to a list of full axis names
    (`["anterior", "posterior", "superior"]`).
space_is_compatible
    Check whether two standard spaces can map to/from each other.
space_to_name
    Get neurospace name from spatial axis names.


Attributes
----------
default: ng.CoordinateSpace
    Default display space (`"xyz"`).
defaultnames : list[str]
    List of all default orientations (`"xyz"`, `"zyx"`, etc).
defaultspaces : dict[str, ng.CoordinateSpace]
    Mapping to all neuroglancer standard spaces (xyz, zyx, etc).
letter2full : dict[str, str]
    Mapping from short to long axis names.
neuronames : list[str]
    List of all existing neuroimaging orientations (`"ras"`, `"lpi"`, etc).
neurospaces : dict[str, ng.CoordinateSpace]
    Mapping to all known neuroimaging-oriented spaced (RAS, LPI, etc),
    as well as all neuroglancer standard spaces (xyz, zyx, etc).
neurotransforms : dict[tuple[str, str], ng.CoordinateSpaceTransform]
    Mapping to transforms between neuroimaging spaces.
    The keys are either pairs of `str`, or pairs of `ng.CoordinateSpace`
    instances from the `neurospaces` dictionnary.

"""
# stdlib
import itertools

# externals
import neuroglancer as ng
import numpy as np

# internals
import ngtools.units as U

letter2full: dict[str, str] = {
    "r": "right",
    "l": "left",
    "a": "anterior",
    "p": "posterior",
    "i": "inferior",
    "s": "superior",
}


def name_compact2full(name: str) -> list[str]:
    """
    Convert compact axes name to long list of names.

    Examples
    --------
    `compact2full("ras") -> ["right", "anterior", "posterior"]`
    `compact2full("xyz") -> ["x", "y", "z"]`
    """
    if isinstance(name, (list, tuple)):
        return name
    name = list(name.lower())
    return [letter2full.get(n, n) for n in name]


def _get_neuronames(ndim: int | None = None) -> list[str]:
    """
    Return all short neuronames ("ras", "lpi", etc).

    By default, return names for all possible number of dimensions (1, 2, 3).
    Specify `ndim` to only return names for a specific number of dimensions.
    """
    if ndim is None:
        range_letters = list(range(1, 4))
    else:
        range_letters = [ndim]
    all_neuronames = []
    all_letters = (("r", "l"), ("a", "p"), ("i", "s"))
    for ndim in range_letters:
        for letters in itertools.combinations(all_letters, ndim):
            for perm in itertools.permutations(range(ndim)):
                for flip in itertools.product([0, 1], repeat=ndim):
                    space = [letters[p][f] for p, f in zip(perm, flip)]
                    space = "".join(space)
                    all_neuronames.append(space)
    return all_neuronames


def _get_defaultnames(ndim: int | None = None) -> list[str]:
    """
    Return all short default names ("xyz", "zyx", etc).

    By default, return names for all possible number of dimensions (1, 2, 3).
    Specify `ndim` to only return names for a specific number of dimensions.
    """
    if ndim is None:
        range_letters = list(range(1, 4))
    else:
        range_letters = [ndim]
    defaultnames = []
    for ndim in range_letters:
        for letters in itertools.combinations(["x", "y", "z"], ndim):
            for name in itertools.permutations(letters):
                defaultnames.append("".join(name))
    return defaultnames


neuronames: list[str] = _get_neuronames()
defaultnames: list[str] = _get_defaultnames()


def _get_src2ras(src: str) -> np.ndarray:
    """Return a matrix mapping from `src` to RAS space."""
    xyz2ras = {"x": "r", "y": "a", "z": "s"}
    src = [xyz2ras.get(x, x) for x in src.lower()]
    pos = list("ras")
    neg = list("lpi")
    mat = np.eye(4)
    # permutation
    perm = [pos.index(x) if x in pos else neg.index(x) for x in src]
    perm += [3]
    mat = mat[:, perm]
    # flip
    flip = [-1 if x in neg else 1 for x in src]
    flip += [1]
    mat *= np.asarray(flip)
    return mat


def _get_src2dst(src: str, dst: str) -> np.ndarray:
    """Return a matrix mapping from `src` to `dst` space."""
    src_to_ras = _get_src2ras(src)
    dst_to_ras = _get_src2ras(dst)
    return np.linalg.pinv(dst_to_ras) @ src_to_ras


# Build coordinate spaces
neurospaces: dict[str, ng.CoordinateSpace] = {
    name: ng.CoordinateSpace({
        letter2full[letter]: [1, "mm"]
        for letter in name
    })
    for name in neuronames
}

defaultspaces: dict[str, ng.CoordinateSpace] = {
    name: ng.CoordinateSpace({
        letter: [1, "mm"]
        for letter in name
    })
    for name in defaultnames
}

default = defaultspaces["xyz"]
neurospaces.update(defaultspaces)


def space_is_compatible(src: str, dst: str) -> bool:
    """
    Check whether two spaces can map to/from each other.

    True if both spaces describe the same set of 3 axes.
    """
    to_xyz = {
        "r": "x",
        "l": "x",
        "a": "y",
        "p": "y",
        "s": "z",
        "i": "z",
    }
    src = list(map(lambda x: to_xyz.get(x, x), src))
    dst = list(map(lambda x: to_xyz.get(x, x), dst))
    for x in src:
        if x not in dst:
            return False
    for x in dst:
        if x not in src:
            return False
    return True


_NeuroTransformMap = dict[tuple[str, str], ng.CoordinateSpaceTransform]


def _make_neurotransforms() -> _NeuroTransformMap:
    # Build coordinate transforms
    # 1) key = string
    neurotransforms: dict[tuple[str, str], ng.CoordinateSpaceTransform] = {
        (src, dst): ng.CoordinateSpaceTransform(
            matrix=_get_src2dst(src, dst)[:3, :4],
            input_dimensions=neurospaces[src],
            output_dimensions=neurospaces[dst],
        )
        for src in neuronames
        for dst in neuronames
        if len(src) == len(dst) and space_is_compatible(src, dst)
    }
    # 2) key = CoordinateSpace instances
    neurotransforms.update({
        (neurospaces[src], neurospaces[dst]): neurotransforms[src, dst]
        for src in neuronames
        for dst in neuronames
        if len(src) == len(dst) and space_is_compatible(src, dst)
    })
    # 3) neurospace to/from xyz
    for name in defaultnames:
        ndim = len(name)
        for neuroname in _get_neuronames(ndim):
            if not space_is_compatible(name, neuroname):
                continue
            matrix = _get_src2dst(neuroname, name)
            neurotransforms[(neuroname, name)] \
                = neurotransforms[(neurospaces[neuroname],
                                   defaultspaces[name])] \
                = ng.CoordinateSpaceTransform(
                    matrix=matrix[:ndim, :ndim+1],
                    input_dimensions=neurospaces[neuroname],
                    output_dimensions=defaultspaces[name],
                )
            neurotransforms[(name, neuroname)] \
                = neurotransforms[(defaultspaces[name],
                                   neurospaces[neuroname])] \
                = ng.CoordinateSpaceTransform(
                    matrix=np.linalg.inv(matrix)[:ndim, :ndim+1],
                    input_dimensions=defaultspaces[name],
                    output_dimensions=neurospaces[neuroname],
                )
    return neurotransforms


neurotransforms: _NeuroTransformMap = _make_neurotransforms()


def convert_space(
    space: ng.CoordinateSpace,
    units: str | list[str] | dict[str | tuple[str], str | list[str]] = "base",
    *,
    names: str | list[str] | None = None,
) -> ng.CoordinateSpace:
    """
    Convert units of a coordinate space.

    !!! example
        === "A"
            __Ensure that all axes have SI unit (without prefix)__
            ```python
            new_space = convert_space(space)
            # or
            new_space = convert_space(space, "base")
            # or
            new_space = convert_space(space, ["m", "s", "hz", "rad/s"])
            # or
            new_space = normalize_space(space)
            ```
        === "B"
            __Ensure that spatial axes have millimeter unit__
            ```python
            new_space = convert_space(space, "mm")
            ```
        === "C"
            __Ensure that specific axes have specific unit__
            ```python
            new_space = convert_space(space, {"x": "mm", "y": "mm", "t": "ms"})
            # or
            new_space = convert_space(space, {("x", "y"): "mm", "t": "ms"})
            # or
            new_space = convert_space(space, {("x", "y", "t"): ["mm", "ms"]})
            ```

    !!! info "See also: `normalize_space`."

    Parameters
    ----------
    space : ng.CoordinateSpace
        Coordinate space to convert.
    units : str | list[str] | dict[str | tuple[str], str | list[str]]
        Output unit(s).

        * If `"base"`, convert all units to their zero-exponent basis
        (`{"m", "s", "hz", "rad/s"}`).
        * If a dictionary, it should map axis name(s) to target unit(s).
    names : str | list[str] | dict[str, str] | None
        Name(s) of axis to convert.
        If `None`, all axes that have compatible units.
        Cannot be used if `units` is a dictionary.

    Returns
    -------
    space : ng.CoordinateSpace
        Converted coordinate space.
    """
    inp = space.to_json()

    # Format units
    if units == "base":
        units = ["m", "s", "hz", "rad/s"]
    if isinstance(units, str):
        units = [units]

    # Build map from input axes to possible units
    # -> dict[names: tuple[str], units: list[str]]
    if isinstance(units, dict):
        if names is not None:
            raise ValueError("Cannot use both `names` and `unit` dictionary.")
        names_to_units = {
            _ensure_tuple(name): _ensure_list(unit)
            for name, unit in units.items()
        }
    else:
        units = _ensure_list(units)
        kinds = [U.split_unit(unit)[1] for unit in units]
        if names is None:
            names = [
                name
                for name, (_, unit) in inp.items()
                if U.split_unit(unit)[1] in kinds
            ]
        names_to_units = {_ensure_tuple(name): units for name in names}

    # Select only matching units
    # -> dict[name: str, unit: str]
    name_to_unit = {}
    for possible_names, possible_units in names_to_units.items():
        for name in possible_names:
            unit = inp[name][1]
            possible_units = _ensure_list(possible_units)
            for possible_unit in possible_units:
                if U.same_unit_kind(unit, possible_unit):
                    unit = possible_unit
                    break
            name_to_unit[name] = unit

    # Go through each axis and convert if needed.
    out = {}
    for name, (inp_scle, inp_unit) in inp.items():

        # Axis not in map -> keep as is
        if (name not in name_to_unit.keys()):
            out[name] = [inp_scle, inp_unit]
            continue

        # Split and check units are compatible
        out_unit = name_to_unit[name]
        inp_prefix, inp_suffix = U.split_unit(inp_unit)
        out_prefix, out_suffix = U.split_unit(out_unit)
        if inp_suffix != out_suffix:
            raise ValueError(
                f'Incompatible units: "{inp_unit}" and "{out_unit}"')

        # Compute and relative scale
        relprefix = (
            U.SI_PREFIX_EXPONENT[inp_prefix] -
            U.SI_PREFIX_EXPONENT[out_prefix]
        )
        out[name] = [inp_scle * 10**relprefix, out_unit]

    return ng.CoordinateSpace(out)


def normalize_space(space: ng.CoordinateSpace) -> ng.CoordinateSpace:
    """
    Ensure that dimensions have SI units (without prefix).

    !!! info "See also: `convert_space`."
    """
    return convert_space(space, "base")


def space_to_name(
    space: ng.CoordinateSpace,
    compact: bool = False,
) -> str | list[str]:
    """Get spatial axis name from coordinate space."""
    space = space.to_json().items()
    space = [
        name for name, (_, unit) in space
        if U.split_unit(unit)[-1][-1:] == "m"
    ]
    if compact:
        space = "".join([name[0].lower() for name in space])
    return space


def _ensure_list(x: object, n: int | None = None, **kwargs) -> list:
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if not isinstance(x, (list, tuple)):
        x = [x]
    x = list(x)
    if n is not None:
        if len(x) < n:
            default = kwargs.get("default", x[-1])
            x += [default] * (n - len(x))
        elif len(x) > n:
            x = x[:n]
    return x


def _ensure_tuple(x: object, n: int | None = None, **kwargs) -> tuple:
    return tuple(_ensure_list(x, n, **kwargs))
