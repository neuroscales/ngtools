"""
Utilities to manipulate spaces (`ng.CoordinateSpace`) and spatial
transforms (`ng.CoordinateSpaceTransform`) in neuroglancer.

Functions
---------
compose
    Compose two `ng.CoordinateSpaceTransform`.
convert
    Convert the units of a transform, while keeping the tranform equivalent.
ensure_same_scaling
    Convert the units of a transform, so that all axes have the same
    unit and scale, while keeping the tranform equivalent
subtransform
    Keep only axes whose units are of a given kind
to_square
    Ensure that an affine matrix is square (with its homogeneous row).
split_unit
    Split the prefix and unit type of a neuroglancer unit.
si_convert
    Convert a value or list of values between units.
compact2full
    Convert a compact axis name (`'RAS'`) to a list of full axis names
    (`['anterior', 'posterior', 'superior']`).

Attributes
----------
neuronames : list[str]
    List of all existing neuroimaging orientations (RAS, LPI, etc).
neurospaces : dict[str, ng.CoordinateSpace]
    Mapping to all known neuroimaging-oriented spaced (RAS, LPI, etc),
    as well as all neuroglancer standard spaces (xyz, zyx, etc).
neurotransforms : dict((str, str), ng.CoordinateSpaceTransform)
    Mapping to transforms between neuroimaging spaces.
    The keys are either pairs of `str`, or pairs of `ng.CoordinateSpace`
    instances from the `neurospaces` dictionnary.
letter2full : dict[str, str]
    A mapping from single letter neuroaxis name (e.g. `"r"`) to full
    neuroaxis name (e.g. `"right"`)
si_units : list[str]
    Neuroglancer unit types.
si_prefixes : dict[str, int]
    Mapping from neuroglancer unit prefix to the corresponding
    base 10 exponent.
letter2full : dict[str, str]
    Mapping from short axis names to long axis names.
"""
__all__ = [
    'to_square',
    'split_unit',
    'si_prefixes',
    'si_units',
    'letter2full',
    'compact2full',
    'neuronames',
    'neurospaces',
    'neurotransforms',
    'compose',
    'convert',
    'ensure_same_scaling',
    'subtransform',
]
import neuroglancer as ng
import numpy as np
import itertools
import math
from typing import Sequence


def to_square(affine):
    """Ensure an affine matrix is in square (homogeneous) form"""
    if affine.shape[0] == affine.shape[1]:
        return affine
    new_affine = np.eye(affine.shape[-1])
    new_affine[:-1, :] = affine
    return new_affine


mu1 = '\u00B5'
mu2 = '\u03BC'
si_prefixes = {
    "Y": 24,
    "Z": 21,
    "E": 18,
    "P": 15,
    "T": 12,
    "G": 9,
    "M": 6,
    "k": 3,
    "h": 2,
    "": 0,
    "c": -2,
    "m": -3,
    "u": -6,
    mu1: -6,
    mu2: -6,
    "n": -9,
    "p": -12,
    "f": -15,
    "a": -18,
    "z": -21,
    "y": -24,
}

si_units = ["m", "s", "rad/s", "Hz"]


def split_unit(unit):
    """
    Split prefix and unit

    Parameters
    ----------
    unit : [sequence of] str

    Returns
    -------
    prefix : [sequence of] str
    unit : [sequence of] str
    """
    if isinstance(unit, (list, tuple)):
        unit = type(unit)(map(split_unit, unit))
        prefix = type(unit)(map(lambda x: x[0], unit))
        unit = type(unit)(map(lambda x: x[1], unit))
        return prefix, unit

    for strippedunit in ("rad/s", "Hz", "m", "s"):
        if unit.endswith(strippedunit):
            return unit[:-len(strippedunit)], strippedunit
    if not unit:
        return '', ''
    raise ValueError(f'Unknown unit "{unit}"')


def si_convert(x, src, dst):
    """Convert a value or list of values from one unit to another"""
    if isinstance(x, (list, tuple)):
        src = ensure_list(src, len(x))
        dst = ensure_list(dst, len(x))
        return type(x)(map(lambda args: si_convert(*args), zip(x, src, dst)))

    src_prefix, src_unit = split_unit(src)
    dst_prefix, dst_unit = split_unit(dst)
    if src_unit != dst_unit:
        raise ValueError('Cannot convert between different kinds')
    return x * 10 ** (si_prefixes[src_prefix] - si_prefixes[dst_prefix])


letter2full = {
    'r': 'right',
    'l': 'left',
    'a': 'anterior',
    'p': 'posterior',
    'i': 'inferior',
    's': 'superior',
}


def compact2full(name):
    """
    Convert compact axes name to long list of names

    Examples
    --------
    `compact2full('ras') -> ['right', 'anterior', 'posterior']`
    `compact2full('xyz') -> ['x', 'y', 'z']`
    """
    if isinstance(name, Sequence):
        return name
    name = list(name.lower())
    return [letter2full.get(n, n) for n in name]


def get_neuronames(ndim=None):
    """
    Return all short neuronames ("ras", "lpi", etc)

    By default, return names for all possible number of dimensions (1, 2, 3).
    Specify `ndim` to pnly return names for a specific number of dimensions.
    """
    if ndim is None:
        range_letters = list(range(1, 4))
    else:
        range_letters = [ndim]
    all_neuronames = []
    all_letters = (('r', 'l'), ('a', 'p'), ('i', 's'))
    for ndim in range_letters:
        for letters in itertools.combinations(all_letters, ndim):
            for perm in itertools.permutations(range(ndim)):
                for flip in itertools.product([0, 1], repeat=ndim):
                    space = [letters[p][f] for p, f in zip(perm, flip)]
                    space = ''.join(space)
                    all_neuronames.append(space)
    return all_neuronames


def get_defaultnames(ndim=None):
    """
    Return all short default names ("xyz", "zyx", etc)

    By default, return names for all possible number of dimensions (1, 2, 3).
    Specify `ndim` to pnly return names for a specific number of dimensions.
    """
    if ndim is None:
        range_letters = list(range(1, 4))
    else:
        range_letters = [ndim]
    defaultnames = []
    for ndim in range_letters:
        for letters in itertools.combinations(['x', 'y', 'z'], ndim):
            for name in itertools.permutations(letters):
                defaultnames.append(''.join(name))
    return defaultnames


neuronames = get_neuronames()
defaultnames = get_defaultnames()


def _get_src2ras(src):
    """Return a matrix mapping from `src` to RAS space."""
    xyz2ras = {'x': 'r', 'y': 'a', 'z': 's'}
    src = [xyz2ras.get(x, x) for x in src.lower()]
    pos = list('ras')
    neg = list('lpi')
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


def _get_src2dst(src, dst):
    """Return a matrix mapping from `src` to `dst` space."""
    src_to_ras = _get_src2ras(src)
    dst_to_ras = _get_src2ras(dst)
    return np.linalg.pinv(dst_to_ras) @ src_to_ras


# Build coordinate spaces
neurospaces = {
    name: ng.CoordinateSpace(
        names=[letter2full[letter.lower()] for letter in name],
        scales=[1]*len(name),
        units=['mm']*len(name),
    )
    for name in neuronames
}

defaultspaces = {
    name: ng.CoordinateSpace(
        names=list(name),
        scales=[1]*len(name),
        units=['mm']*len(name),
    )
    for name in defaultnames
}
default = defaultspaces['xyz']
neurospaces.update(defaultspaces)


def space_is_compatible(src, dst):
    to_xyz = {
        'r': 'x',
        'l': 'x',
        'a': 'y',
        'p': 'y',
        's': 'z',
        'i': 'z',
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


# Build coordinate transforms
# 1) key = string
neurotransforms = {
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
    for neuroname in get_neuronames(ndim):
        if not space_is_compatible(name, neuroname):
            continue
        matrix = _get_src2dst(neuroname, name)
        neurotransforms[(neuroname, name)] \
            = neurotransforms[(neurospaces[neuroname], defaultspaces[name])] \
            = ng.CoordinateSpaceTransform(
                matrix=matrix[:ndim, :ndim+1],
                input_dimensions=neurospaces[neuroname],
                output_dimensions=defaultspaces[name],
            )
        neurotransforms[(name, neuroname)] \
            = neurotransforms[(defaultspaces[name], neurospaces[neuroname])] \
            = ng.CoordinateSpaceTransform(
                matrix=np.linalg.inv(matrix)[:ndim, :ndim+1],
                input_dimensions=defaultspaces[name],
                output_dimensions=neurospaces[neuroname],
            )


def compose(left, right):
    """
    Compose two transforms

    Parameters
    ----------
    left : ng.CoordinateSpaceTransform
    right : ng.CoordinateSpaceTransform

    Returns
    -------
    composition : ng.CoordinateSpaceTransform
    """

    left = convert(
        left,
        unit=right.output_dimensions.units,
        scale=right.output_dimensions.scales,
        name=right.output_dimensions.names,
        space='input',
    )

    input_dimensions = right.input_dimensions
    output_dimensions = left.output_dimensions
    right_dimensions = right.output_dimensions
    left_dimensions = left.input_dimensions

    onames = output_dimensions.names
    ounits = output_dimensions.units
    oscles = output_dimensions.scales
    inames = input_dimensions.names
    iunits = input_dimensions.units
    iscles = input_dimensions.scales
    lnames = left_dimensions.names
    lunits = left_dimensions.units
    lscles = left_dimensions.scales
    rnames = right_dimensions.names
    runits = right_dimensions.units
    rscles = right_dimensions.scales

    lmatrix = left.matrix
    if lmatrix is None:
        lmatrix = np.eye(len(onames)+1)[:-1]
    lmatrix = to_square(lmatrix)
    rmatrix = right.matrix
    if rmatrix is None:
        rmatrix = np.eye(len(inames)+1)[:-1]
    rmatrix = to_square(rmatrix)

    # delete output axes that do not exist in the input space
    delnames = [n for n in lnames if n not in rnames]
    odel = [onames.index(n) for n in delnames]
    ldel = [lnames.index(n) for n in delnames]
    lkeep = [i for i in range(len(lnames)) if i not in ldel]
    okeep = [i for i in range(len(onames)) if i not in odel]
    lnames = [lnames[i] for i in lkeep]
    lunits = [lunits[i] for i in lkeep]
    lscles = [lscles[i] for i in lkeep]
    onames = [onames[i] for i in okeep]
    ounits = [ounits[i] for i in okeep]
    oscles = [oscles[i] for i in okeep]
    lmatrix = lmatrix[okeep + [-1], :][:, lkeep + [-1]]

    # add axes that exist in the input space but not in the output space
    extnames = [n for n in rnames if n not in lnames]
    onames += extnames
    ounits += [runits[rnames.index(n)] for n in extnames]
    oscles += [rscles[rnames.index(n)] for n in extnames]
    lnames += extnames
    lunits += [runits[rnames.index(n)] for n in extnames]
    lscles += [rscles[rnames.index(n)] for n in extnames]
    lmatrix0 = lmatrix
    lmatrix = np.eye(len(onames)+1)
    n0 = len(lmatrix0)-1
    # copy compatible part
    lmatrix[:n0, :n0] = lmatrix0[:-1, :-1]
    lmatrix[:n0, -1] = lmatrix0[:-1, -1]
    # copy extra part
    rsub = [rnames.index(n) for n in extnames]
    isub = [inames.index(n) for n in extnames]
    lmatrix[n0:-1, n0:-1] = rmatrix[rsub, :][:, isub]
    lmatrix[n0:-1, -1] = rmatrix[rsub, -1]
    # reorder right side
    lmatrix = lmatrix[:, [lnames.index(n) for n in rnames] + [-1]]

    matrix = lmatrix @ rmatrix

    T = ng.CoordinateSpaceTransform(
        input_dimensions=ng.CoordinateSpace(
            names=inames,
            units=iunits,
            scales=iscles,
        ),
        output_dimensions=ng.CoordinateSpace(
            names=onames,
            units=ounits,
            scales=oscles,
        ),
        matrix=matrix[:-1],
    )
    return T


def convert(transform, unit, scale=None, name=None, space='input+output'):
    """
    Convert units of a transform

    Parameters
    ----------
    transform : ng.CoordinateSpaceTransform
        Transform
    unit : [list of] str
        New units
    name : [list of] str, optional
        Name of axis (or axes) to convert.
        Default: all that have compatible units.
    space : {'input', 'output', 'input+output'}
        Space to convert

    Returns
    -------
    transform : ng.CoordinateSpaceTransform
    """
    prefix, unit = split_unit(ensure_list(unit))
    scale = ensure_list(scale)

    matrix = transform.matrix
    if matrix is None:
        matrix = np.eye(len(transform.output_dimensions.names)+1)[:-1]
    matrix = to_square(np.copy(matrix))

    new_transform = ng.CoordinateSpaceTransform(
        matrix=matrix,
        input_dimensions=transform.input_dimensions,
        output_dimensions=transform.output_dimensions,
    )

    for convert_space in ('input', 'output'):
        if convert_space not in space:
            continue

        dimensions = getattr(transform, convert_space + '_dimensions')
        if space == 'output':
            matrix = matrix.T

        dnames = dimensions.names
        dunits = dimensions.units
        dscles = dimensions.scales

        convert_names = name
        if convert_names is None:
            convert_names = [dnames[i] for i in range(len(dnames))
                             if split_unit(dunits[i])[1] in unit]
        convert_names = ensure_list(convert_names)
        convert_units = ensure_list(unit, len(convert_names))
        convert_prfix = ensure_list(prefix, len(convert_names))
        convert_scles = ensure_list(scale, len(convert_names))

        new_units = []
        new_scles = []
        for i, (dunit, dscle, dname) in enumerate(zip(dunits, dscles, dnames)):
            if (dnames[i] not in convert_names) or not dunit:
                new_units.append(dunit)
                new_scles.append(dscle)
                continue
            j = convert_names.index(dname)
            dprefix, dunit = split_unit(dunit)
            new_prefix, new_unit = convert_prfix[j], convert_units[j]
            if dunit != new_unit:
                raise ValueError(
                    f'Incompatible units: "{dunit}" and "{new_unit}"')
            relprefix = si_prefixes[dprefix] - si_prefixes[new_prefix]
            new_scle = dscle * 10**relprefix
            if convert_scles[j] is not None:
                matrix[:, i] *= new_scle / convert_scles[j]
                new_scle = convert_scles[j]
            new_scles.append(new_scle)
            new_units.append(new_prefix + new_unit)

        if space == 'output':
            matrix = matrix.T
        setattr(
            new_transform, convert_space + '_dimensions',
            ng.CoordinateSpace(
                names=dnames,
                units=new_units,
                scales=new_scles,
            )
        )

    new_transform.matrix = matrix
    return new_transform


def ensure_same_scaling(transform):
    """
    Ensure that all axes and both input and output spaces use the same
    unit and the same scaling.
    """
    unitmap = {}
    scalemap = {}

    # find all scales and prefixes in the input/output spaces
    for dimensions in (transform.outputDimensions, transform.inputDimensions):
        for unit, scale in zip(dimensions.units, dimensions.scales):
            prefix, unit = split_unit(unit)
            logprefix = si_prefixes[prefix]
            logprefix += int(round(math.log10(scale)))
            unitmap.setdefault(unit, [])
            unitmap[unit].append(logprefix)

    # choose the unit and scale that are closest/most common
    for unit, logprefixes in dict(unitmap).items():
        logprefix, count = None, 0
        for p in set(logprefixes):
            if logprefixes.count(p) > count:
                logprefix = p
        prefix, dist = None, float('inf')
        for p, logp in si_prefixes.items():
            if abs(logprefix - logp) < abs(dist):
                dist = logprefix - logp
                prefix = p
        unitmap[unit] = prefix + unit
        scalemap[unit] = 10 ** dist

    # convert the transform
    for unit in unitmap.keys():
        transform = convert(transform, unitmap[unit], scalemap[unit])
    return transform


def subtransform(transform, unit):
    """
    Generate a subtransform by keeping only axes whose unit os of kind `unit`
    """
    kind = split_unit(unit)[1]
    idim = transform.input_dimensions
    odim = transform.output_dimensions
    matrix = transform.matrix
    if matrix is None:
        matrix = np.eye(len(idim.names)+1)[:-1]
    # filter input axes
    ikeep = []
    for i, unit in enumerate(idim.units):
        unit = split_unit(unit)[1]
        if unit == kind:
            ikeep.append(i)
    idim = ng.CoordinateSpace(
        names=[idim.names[i] for i in ikeep],
        scales=[idim.scales[i] for i in ikeep],
        units=[idim.units[i] for i in ikeep],
    )
    matrix = matrix[:, ikeep + [-1]]
    # filter output axes
    okeep = []
    for i, unit in enumerate(odim.units):
        unit = split_unit(unit)[1]
        if unit == kind:
            okeep.append(i)
    odim = ng.CoordinateSpace(
        names=[odim.names[i] for i in okeep],
        scales=[odim.scales[i] for i in okeep],
        units=[odim.units[i] for i in okeep],
    )
    matrix = matrix[okeep, :]

    return ng.CoordinateSpaceTransform(
        matrix=matrix,
        input_dimensions=idim,
        output_dimensions=odim,
    )


def ensure_list(x, n=None, **kwargs):
    if isinstance(x, np.ndarray):
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
