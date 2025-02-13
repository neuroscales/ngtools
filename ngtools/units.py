"""
Tools to convert between units.

Functions
---------
split_unit
    Split prefix and unit
convert_unit
    Convert a value between units
normalize_unit
    Convert to prefixless SI unit
unit_to_scale
    Return the scale of a arbitrary unit with respect to its SI equivalent
as_ome_unit
    Find OME version of a unit
as_nifti_unit
    Find NIfTI version of a unit
as_neuroglancer_unit
    Find Neuroglancer version of a unit (= short SI)

Attributes
----------
UnitDomain
    Type hint for unit domains, _i.e._, `Literal["space", "time", "all"]`
OME_UNITS : dict[UnitDomain, list[str]]
    Dictionary where values are lists of valid OME units.
NIFTI_UNITS : dict[UnitDomain, list[str]]
    Dictionary where values are lists of valid NIfTI units.
SI_UNITS_SHORT : dict[UnitDomain, list[str]]
    Dictionary where values are lists of valid Neuroglancer (short SI) units.
SI_UNITS_LONG : dict[UnitDomain, list[str]]
    Dictionary where values are lists of valid SI (long) units.
SI_PREFIX_SHORT2LONG : dict[str, str]
    Dictionary that maps short forms of SI prefixes to long forms
SI_PREFIX_LONG2SHORT : dict[str, str]
    Dictionary that maps long forms of SI prefixes to short forms
SI_PREFIX_EXPONENT : dict[str, float]
    Dictionary that maps short forms of SI prefixes to their exponent value
UNIT_SHORT2LONG : dict[UnitDomain, dict[str, str]]
    Dictionary where values are dictionaries that map short forms of a
    unit to long forms
UNIT_LONG2SHORT : dict[UnitDomain, dict[str, str]]
    Dictionary where values are dictionaries that map long forms of a
    unit to short forms
"""
# stdlib
from typing import Literal

# externals
from numpy.typing import ArrayLike

_MU1 = '\u00B5'
_MU2 = '\u03BC'

UnitDomain = Literal["space", "time", "frequency", "angular_frequency", "all"]
UNIT_DOMAINS = UnitDomain.__args__

# ======================================================================
#   OME-NGFF
# ======================================================================

OME_UNITS: dict[UnitDomain, list[str]] = {
    "space": [
        "angstrom",
        "attometer",
        "centimeter",
        "decimeter",
        "exameter",
        "femtometer",
        "foot",
        "gigameter",
        "hectometer",
        "inch",
        "kilometer",
        "megameter",
        "meter",
        "micrometer",
        "mile",
        "millimeter",
        "nanometer",
        "parsec",
        "petameter",
        "picometer",
        "terameter",
        "yard",
        "yoctometer",
        "yottameter",
        "zeptometer",
        "zettameter",
    ],
    "time": [
        "attosecond",
        "centisecond",
        "day",
        "decisecond",
        "exasecond",
        "femtosecond",
        "gigasecond",
        "hectosecond",
        "hour",
        "kilosecond",
        "megasecond",
        "microsecond",
        "millisecond",
        "minute",
        "nanosecond",
        "petasecond",
        "picosecond",
        "second",
        "terasecond",
        "yoctosecond",
        "yottasecond",
        "zeptosecond",
        "zettasecond",
    ]
}
OME_UNITS["all"] = [unit for units in OME_UNITS.values() for unit in units]

# ======================================================================
#   NIFTI
# ======================================================================

NIFTI_UNITS: dict[UnitDomain, list[str]] = {
    "space": [
        "meter",
        "mm",
        "micron",
    ],
    "time": [
        "sec",
        "msec",
        "usec",
    ],
    "frequency": [
        "hz",
    ],
    "angular_frequency": [
        "rads",
        "rad/s",
    ]
}
NIFTI_UNITS["all"] = [unit for units in OME_UNITS.values() for unit in units]
NIFTI_UNITS["all"] += ["unknown", "ppm"]

# ======================================================================
#   SI PREFIXES
# ======================================================================

SI_PREFIX_SHORT2LONG: dict[str, str] = {
    "Q": "quetta",
    "R": "ronna",
    "Y": "yotta",
    "Z": "zetta",
    "E": "exa",
    "P": "peta",
    "T": "tera",
    "G": "giga",
    "M": "mega",
    "K": "kilo",
    "k": "kilo",
    "H": "hecto",
    "h": "hecto",
    "D": "deca",
    "da": "deca",
    "": "",
    "d": "deci",
    "c": "centi",
    "m": "milli",
    "u": "micro",
    _MU1: "micro",
    _MU2: "micro",
    "n": "nano",
    "p": "pico",
    "f": "femto",
    "a": "atto",
    "z": "zepto",
    "y": "yocto",
    "r": "ronto",
    "q": "quecto",
}

SI_PREFIX_LONG2SHORT: dict[str, str] = {
    long: short
    for short, long in SI_PREFIX_SHORT2LONG.items()
}

SI_PREFIX_EXPONENT: dict[str, int] = {
    "Q": 30,
    "R": 27,
    "Y": 24,
    "Z": 21,
    "E": 18,
    "P": 15,
    "T": 12,
    "G": 9,
    "M": 6,
    "K": 3,
    "k": 3,
    "H": 2,
    "h": 2,
    "D": 1,
    "da": 1,
    "": 0,
    "d": -1,
    "c": -2,
    "m": -3,
    "u": -6,
    _MU1: -6,
    _MU2: -6,
    "n": -9,
    "p": -12,
    "f": -15,
    "a": -18,
    "z": -21,
    "y": -24,
    "r": -27,
    "q": -30,
}

# ======================================================================
#   SI/NEUROGLANCER UNITS
# ======================================================================

UNIT_SHORT2LONG: dict[UnitDomain, dict[str, str]] = {}
"""Mapping from short name to long name (per domain)."""

UNIT_LONG2SHORT: dict[UnitDomain, dict[str, str]] = {}
"""Mapping from long name to short name (per domain)."""

UNIT_SCALE: dict[UnitDomain, dict[str, float]] = {}
"""Mapping from short name to scale (per domain)."""

SI_UNITS_SHORT: dict[UnitDomain, list[str]] = {}
"""List of short SI units (per category)."""

SI_UNITS_LONG: dict[UnitDomain, list[str]] = {}
"""List of long SI units (per domain)."""

# ----------------------------------------------------------------------
#   SPACE
# ----------------------------------------------------------------------

# SI
UNIT_SHORT2LONG["space"] = {
    short + "m": long + "meter"
    for short, long in SI_PREFIX_SHORT2LONG.items()
}
UNIT_SCALE["space"] = {
    prefix + "m": 10**exponent
    for prefix, exponent in SI_PREFIX_EXPONENT.items()
}

SI_UNITS_SHORT["space"] = list(UNIT_SHORT2LONG["space"].keys())
SI_UNITS_LONG["space"] = list(UNIT_SHORT2LONG["space"].values())

# Non-SI
UNIT_SHORT2LONG["space"].update({
    "mi": "mile",
    "yd": "yard",
    "ft": "foot",
    "in": "inch",
    """: "foot",
    """: "inch",
    "Å": "angstrom",
    "pc": "parsec",
})
UNIT_SCALE["space"].update({
    "mi": 1609.344,
    "yd": 0.9144,
    "ft": 0.3048,
    "'": 0.3048,
    "in": 25.4E-3,
    '"': 25.4E-3,
    "Å": 1E-10,
    "pc": 3.0857E16,
})

# Inverse map
UNIT_LONG2SHORT["space"] = {
    long: short
    for short, long in UNIT_SHORT2LONG["space"].items()
}
UNIT_LONG2SHORT["space"]["micron"] = "u"

# ----------------------------------------------------------------------
#   TIME
# ----------------------------------------------------------------------

# SI
UNIT_SHORT2LONG["time"] = {
    short + "s": long + "second"
    for short, long in SI_PREFIX_SHORT2LONG.items()
}
UNIT_SCALE["time"] = {
    prefix + "s": 10**exponent
    for prefix, exponent in SI_PREFIX_EXPONENT.items()
}

SI_UNITS_SHORT["time"] = list(UNIT_SHORT2LONG["time"].keys())
SI_UNITS_LONG["time"] = list(UNIT_SHORT2LONG["time"].values())

# Non-SI
UNIT_SHORT2LONG["time"].update({
    "y": "year",
    "d": "day",
    "h": "hour",
    "m": "minute",
    "s": "second",
})
UNIT_SCALE["time"].update({
    "y": 365.25*24*60*60,
    "d": 24*60*60,
    "h": 60*60,
    "m": 60,
})

# Inverse map
UNIT_LONG2SHORT["time"] = {
    long: short
    for short, long in UNIT_SHORT2LONG["time"].items()
}


# ----------------------------------------------------------------------
#   FREQUENCY
# ----------------------------------------------------------------------

# Only Hertz is used in neuroglancer and nifti so we only deal with
# this single unit.
# TODO: use a proper unit library to handle conversions?

# SI
UNIT_SHORT2LONG["frequency"] = {
    short + "hz": long + "hertz"
    for short, long in SI_PREFIX_SHORT2LONG.items()
}
UNIT_SHORT2LONG["frequency"].update({
    short + "Hz": long + "hertz"
    for short, long in SI_PREFIX_SHORT2LONG.items()
})
UNIT_SCALE["frequency"] = {
    prefix + "hz": 10**exponent
    for prefix, exponent in SI_PREFIX_EXPONENT.items()
}

SI_UNITS_SHORT["frequency"] = list(UNIT_SHORT2LONG["frequency"].keys())
SI_UNITS_LONG["frequency"] = list(UNIT_SHORT2LONG["frequency"].values())

# Inverse map
UNIT_LONG2SHORT["frequency"] = {
    long: short
    for short, long in UNIT_SHORT2LONG["frequency"].items()
}

# ----------------------------------------------------------------------
#   ANGULAR FREQUENCY
# ----------------------------------------------------------------------

# Only rad/s is used in neuroglancer and nifti so we only deal with
# this single unit.
# TODO: use a proper unit library to handle conversions?

# SI
UNIT_SHORT2LONG["angular_frequency"] = {
    short + "rad/s": long + "radian/second"
    for short, long in SI_PREFIX_SHORT2LONG.items()
}
UNIT_SHORT2LONG["angular_frequency"].update({
    short + "rads": long + "radian/second"
    for short, long in SI_PREFIX_SHORT2LONG.items()
})
UNIT_SCALE["angular_frequency"] = {
    prefix + "rad/s": 10**exponent
    for prefix, exponent in SI_PREFIX_EXPONENT.items()
}

SI_UNITS_SHORT["angular_frequency"] \
    = list(UNIT_SHORT2LONG["angular_frequency"].keys())
SI_UNITS_LONG["angular_frequency"] \
    = list(UNIT_SHORT2LONG["angular_frequency"].values())

# Inverse map
UNIT_LONG2SHORT["angular_frequency"] = {
    long: short
    for short, long in UNIT_SHORT2LONG["angular_frequency"].items()
}


# ----------------------------------------------------------------------
#   ALL
# ----------------------------------------------------------------------


SI_UNITS_SHORT["all"] = [u for units in SI_UNITS_SHORT.values() for u in units]
SI_UNITS_LONG["all"] = [u for units in SI_UNITS_LONG.values() for u in units]


# ----------------------------------------------------------------------
#   PUBLIC FUNCTIONS
# ----------------------------------------------------------------------


def split_unit(unit: str) -> tuple[str, str]:
    """
    Split prefix and unit.

    Parameters
    ----------
    unit : [sequence of] str

    Returns
    -------
    prefix : [sequence of] str
    unit : [sequence of] str
    """
    unit = as_neuroglancer_unit(unit)

    if unit == "":
        return ("", "")

    if isinstance(unit, (list, tuple)):
        unit = type(unit)(map(split_unit, unit))
        prefix = type(unit)(map(lambda x: x[0], unit))
        unit = type(unit)(map(lambda x: x[1], unit))
        return prefix, unit

    for strippedunit in ("rad/s", "hz", "m", "s"):
        if unit.endswith(strippedunit):
            return unit[:-len(strippedunit)], strippedunit
    if not unit:
        return '', ''
    raise ValueError(f'Unknown unit "{unit}"')


def same_unit_kind(src: str, dst: str) -> bool:
    """Check whether two units are of the same kind."""
    src = as_neuroglancer_unit(src)
    dst = as_neuroglancer_unit(dst)
    if src == "" and dst == "":
        return True
    if src == "" or dst == "":
        return False
    return split_unit(src)[1] == split_unit(dst)[1]


def convert_unit(
    value: ArrayLike | float,
    src: str,
    dst: str,
) -> ArrayLike | float:
    """Convert a value between different units.

    Parameters
    ----------
    value : float or ArrayLike
        Value to convert
    src : str
        Source unit
    dst : str
        Destination unit

    Returns
    -------
    value : float or array
        Converted value
    """
    if isinstance(value, (list, tuple)):
        src = ensure_list(src, len(value))
        dst = ensure_list(dst, len(value))
        return type(value)([
            convert_unit(x, s, d)
            for x, s, d in zip(value, src, dst)
        ])
    src = unit_to_scale(src)
    dst = unit_to_scale(dst)
    return value * (src / dst)


def normalize_unit(
    value: ArrayLike | float,
    unit: str
) -> ArrayLike | float:
    """
    Normalize a unit (i.e., convert to prefixless SI unit).

    Parameters
    ----------
    value : float or ArrayLike
        Value to convert
    unit : str
        Source unit

    Returns
    -------
    value : float or array
        Converted value
    """
    kind = split_unit(unit)[1]
    return convert_unit(value, unit, kind)


def ensure_list(x: object, n: int | None = None, **kwargs: dict) -> list:
    """
    Ensure that `x` is a list.

    Arrays are converted to nested lists, whereas scalars are insereted
    into a list of length 1.

    Parameters
    ----------
    x : object
        Input value(s)
    n : int, optional
        Target length of the list.
        If the input is longer, it gets cropped.
        Otherwise, it gets padded with the default value.
    default : any, optional
        Value to use to pad the list.
        If not provided, use last value in the list.

    Returns
    -------
    x : list
    """
    if hasattr(x, 'tolist'):
        x = x.tolist()
    if not isinstance(x, (list, tuple)):
        x = [x]
    x = list(x)
    if n is not None:
        if len(x) < n:
            default = kwargs.get('default', x[-1])
            x += [default] * (n - len(x))
        elif len(x) > n:
            x = x[:n]
    return x


def as_short_unit(unit: str) -> str:
    """
    Return the shortest representation for this unit.

    Parameters
    ----------
    unit : str
        Any unit.

    Returns
    -------
    unit : str
        An short unit.

    Raises
    ------
    ValueError
        If the input unit is unknown.

    """
    if unit == "":
        return unit
    if unit.endswith("sec"):
        # nifti
        unit = unit[:-2]
    for units in UNIT_SHORT2LONG.values():
        if unit in units:
            return unit
    for units in UNIT_LONG2SHORT.values():
        if unit in units:
            return units[unit]
    if unit in SI_PREFIX_SHORT2LONG:
        return unit
    if unit in SI_PREFIX_LONG2SHORT:
        return SI_PREFIX_LONG2SHORT[unit]
    raise ValueError("Unknown unit:", unit)


def as_long_unit(unit: str) -> str:
    """
    Return the long representation for this unit.

    Parameters
    ----------
    unit : str
        Any unit.

    Returns
    -------
    unit : str
        An long unit.

    Raises
    ------
    ValueError
        If the input unit is unknown.

    """
    if unit == "":
        return unit
    if unit.endswith("sec"):
        # nifti
        unit = unit[:-2]
    for units in UNIT_LONG2SHORT.values():
        if unit in units:
            return unit
    for units in UNIT_SHORT2LONG.values():
        if unit in units:
            return units[unit]
    if unit in SI_PREFIX_LONG2SHORT:
        return unit
    if unit in SI_PREFIX_SHORT2LONG:
        return SI_PREFIX_SHORT2LONG[unit]
    raise ValueError("Unknown unit:", unit)


def as_neuroglancer_unit(unit: str) -> str:
    """Return the Neuroglancer representation of a unit.

    Parameters
    ----------
    unit : str
        Any unit.

    Returns
    -------
    unit : str
        An OME-compatible unit.

    Raises
    ------
    ValueError
        If the input unit does not have an exactly equivalent
        representation in OME.
    """
    if unit == "":
        return unit
    short_unit = as_short_unit(unit)
    if short_unit not in SI_UNITS_SHORT["all"]:
        raise ValueError("Input is not a SI unit:", unit)
    # Couple of fixes for non-neuroglancer aliases
    if short_unit.endswith("Hz"):
        short_unit = short_unit[:-2] + "hz"
    if short_unit.endswith("rads"):
        short_unit = short_unit[:-4] + "rad/s"
    if short_unit[:1] in (_MU1, _MU2):
        short_unit = "u" + short_unit[1:]
    return short_unit


def as_ome_unit(unit: str) -> str:
    """Return the OME-compatible representation of a unit.

    Parameters
    ----------
    unit : str
        Any unit

    Returns
    -------
    unit : str
        An OME-compatible unit

    Raises
    ------
    ValueError
        If the input unit does not have an exactly equivalent
        representation in OME.
    """
    if unit == "":
        return unit
    long_unit = as_long_unit(unit)
    if long_unit not in OME_UNITS["all"]:
        raise ValueError("Input is not a OME unit:", unit)
    return long_unit


def as_nifti_unit(unit: str) -> str:
    """
    Return the NIfTI-compatible representation of a unit.

    Parameters
    ----------
    unit : str
        Any unit

    Returns
    -------
    unit : str
        A NIfTI-compatible unit.
        If the unit does not have a NIfTI representation, the NIfTI type
        `"unknown"` is returned.
    """
    if unit == "":
        return "unknown"
    unit = as_ome_unit(unit)
    return {
        "meter": "meter",
        "millimeter": "mm",
        "micrometer": "micron",
        "second": "sec",
        "millisecond": "msec",
        "microsecond": "usec",
    }.get(unit, "unknown")


def unit_to_scale(unit: str) -> float:
    """
    Return the scale of a arbitrary unit with respect to its SI equivalent.

    Parameters
    ----------
    unit : str
        Any unit

    Returns
    -------
    scale : float
        The corresponding SI scale

    Raises
    ------
    ValueError
        If the unit does not have a SI equivalent.
    """
    if unit == "":
        return 1.0
    if unit in UNIT_LONG2SHORT["space"]:
        unit = UNIT_LONG2SHORT["space"][unit]
    elif unit in UNIT_LONG2SHORT["time"]:
        unit = UNIT_LONG2SHORT["time"][unit]
    elif unit in SI_PREFIX_LONG2SHORT:
        unit = SI_PREFIX_LONG2SHORT[unit]
    if unit in UNIT_SCALE["space"]:
        unit = UNIT_SCALE["space"][unit]
    elif unit in UNIT_SCALE["time"]:
        unit = UNIT_SCALE["time"][unit]
    elif unit in SI_PREFIX_EXPONENT:
        unit = 10 ** SI_PREFIX_EXPONENT[unit]
    if isinstance(unit, str):
        raise ValueError("Unknown unit", unit)
    return unit
