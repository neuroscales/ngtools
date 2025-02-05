import numpy as np
from ast import literal_eval
import math as pymath
from .utils import ensure_list

affine_subbasis_choices = ('T', 'R', 'Z', 'Z0', 'I', 'S', 'SC')
affine_subbasis_aliases = dict(
    T=['translation', 'translations'],
    R=['rot', 'rotation', 'rotations'],
    Z=['zoom', 'zooms', 'scale', 'scales', 'scaling', 'scalings'],
    Z0=['isovolume', 'isovolumic'],
    I=['isotropic', 'isozoom', 'isoscale', 'isoscaling'],
    S=['shear', 'shears'],
)


def affine_subbasis(mode, dim=3, sub=None, dtype=None):
    """Generate a basis set for the algebra of some (Lie) group of matrices.

    The basis is returned in homogeneous coordinates, even if
    the group required does not require translations. To extract the
    linear part of the basis: lin = basis[:-1, :-1].

    This function focuses on very simple (and coherent) groups.

    Note that shears generated by the 'S' basis are not exactly the same
    as classical shears ('SC'). Setting one classical shear parameter to
    a non-zero value generally applies a gradient of translations along
    a direction:
    + -- +         + -- +
    |    |   =>   /    /
    + -- +       + -- +
    Setting one Lie shear parameter to a non zero value is more alike
    to performing an expansion in one (diagonal) direction and a
    contraction in the orthogonal (diagonal) direction. It is a bit
    harder to draw in ascii, but it can also be seen as a horizontal
    shear followed by a vertical shear.
    The 'S' basis is orthogonal to the 'R' basis, but the 'SC' basis is
    not.

    Parameters
    ----------
    mode : {'T', 'R', 'Z', 'Z0', 'I', 'S', 'SC'}
        Group that should be encoded by the basis set:
            * 'T'   : Translations                     [dim]
            * 'R'   : Rotations                        [dim*(dim-1)//2]
            * 'Z'   : Zooms (= anisotropic scalings)   [dim]
            * 'Z0'  : Isovolumic scalings              [dim-1]
            * 'I'   : Isotropic scalings               [1]
            * 'S'   : Shears (symmetric)               [dim*(dim-1)//2]
            * 'SC'  : Shears (classic)                 [dim*(dim-1)//2]
        If the group name is appended with a list of integers, they
        have the same use as ``sub``. For example 'R[0]' returns the
        first rotation basis only. This grammar cannot be used in
        conjunction with the ``sub`` keyword.

    dim : int, default=3
        Dimension

    sub : int or list[int], optional
        Request only subcomponents of the basis.

    dtype : torch.type, optional
        Data type of the returned array.


    Returns
    -------
    basis : (F, dim+1, dim+1) tensor[dtype]
        Basis set, where ``F`` is the number of basis functions.

    """

    # Authors
    # -------
    # .. John Ashburner <j.ashburner@ucl.ac.uk> : original Matlab code
    # .. Yael Balbastre <yael.balbastre@gmail.com> : Python code

    # Check if sub passed in mode
    mode = mode.split('[')
    if len(mode) > 1:
        if sub is not None:
            raise ValueError('Cannot use both ``mode`` and ``sub`` '
                             'to specify a sub-basis.')
        sub = '[' + mode[1]
        sub = literal_eval(sub)  # Safe eval for list of native types
    mode = mode[0]

    if mode not in affine_subbasis_choices:
        for key, aliases in affine_subbasis_aliases.items():
            if mode in aliases:
                mode = key
                break
    if mode not in affine_subbasis_choices:
        raise ValueError('mode must be one of {}.'
                         .format(affine_subbasis_choices))

    # Compute the basis

    if mode == 'T':
        basis = np.zeros((dim, dim + 1, dim + 1), dtype=dtype)
        for i in range(dim):
            basis[i, i, dim] = 1

    elif mode == 'Z':
        basis = np.zeros((dim, dim + 1, dim + 1), dtype=dtype)
        for i in range(dim):
            basis[i, i, i] = 1

    elif mode == 'Z0':
        basis = np.zeros((dim - 1, dim + 1), dtype=dtype)
        for i in range(dim -1):
            basis[i, i] = 1
            basis[i, i + 1] = -1
        # Orthogonalise numerically.
        u, s, _ = np.linalg.svd(basis)
        s = s[..., None]
        basis = np.matmul(u.T, basis) / s
        diagbasis = basis
        basis = np.zeros(basis.shape + (dim+1), dtype=dtype)
        basis[..., np.arange(dim+1), np.arange(dim+1)] = diagbasis

        # TODO:
        #   Is there an analytical form?
        #   I would say yes. It seems that we have (I only list the diagonals):
        #       2D: [[a, -a]]           with a = 1/sqrt(2)
        #       3D: [[a, 0, -a],        with a = 1/sqrt(2)
        #            [b, -2*b, b]]           b = 1/sqrt(6)
        #       4D: [[a, -b, b, -a],
        #            [c, -c, -c, c],    with c = 1/sqrt(4)
        #            [b, a, -a, -b]]         a = ?, b = ?

    elif mode == 'I':
        basis = np.eye(dim + 1, dtype=dtype)[None, ...]
        basis /= pymath.sqrt(dim)
        basis[:, dim, dim] = 0

    elif mode == 'R':
        nb_rot = dim * (dim - 1) // 2
        basis = np.zeros((nb_rot, dim + 1, dim + 1), dtype=dtype)
        k = 0
        isqrt2 = 1 / pymath.sqrt(2)
        for i in range(dim):
            for j in range(i + 1, dim):
                basis[k, i, j] = isqrt2
                basis[k, j, i] = -isqrt2
                k += 1

    elif mode == 'S':
        nb_shr = dim * (dim - 1) // 2
        basis = np.zeros((nb_shr, dim + 1, dim + 1), dtype=dtype)
        k = 0
        isqrt2 = 1 / pymath.sqrt(2)
        for i in range(dim):
            for j in range(i + 1, dim):
                basis[k, i, j] = isqrt2
                basis[k, j, i] = isqrt2
                k += 1

    elif mode in ('sc', 'classic shear', 'classic shears'):
        nb_shr = dim * (dim - 1) // 2
        basis = np.zeros((nb_shr, dim + 1, dim + 1), dtype=dtype)
        k = 0
        for i in range(dim):
            for j in range(i + 1, dim):
                basis[k, i, j] = 1
                k += 1

    else:
        # We should never reach this (a test was performed earlier)
        raise ValueError

    # Select subcomponents of the basis
    if sub is not None:
        try:
            sub = list(sub)
        except TypeError:
            sub = [sub]
        basis = basis[sub]

    return basis


affine_basis_choices = ('T', 'SO', 'SE', 'D', 'CSO', 'SL', 'GL+', 'Aff+')
affine_basis_aliases = {
    'T': ['translation', 'translations'],
    'SO': ['rot', 'rotation', 'rotations'],
    'SE': ['rigid'],
    'D': ['dilation', 'dilations'],
    'CSO': ['similitude', 'similitudes'],
    'SL': [],
    'GL+': ['linear'],
    'Aff+': ['affine'],
}
affine_basis_components = {
    'T': ['T'],
    'SO': ['R'],
    'SE': ['T', 'R'],
    'D': ['R', 'I'],
    'CSO': ['T', 'R', 'I'],
    'SL': ['R', 'Z0', 'S'],
    'GL+': ['R', 'Z', 'S'],
    'Aff+': ['T', 'R', 'Z', 'S'],
}


def affine_basis(group='SE', dim=3, dtype=None):
    """Generate a basis set for the algebra of some (Lie) group of matrices.

    The basis is returned in homogeneous coordinates, even if
    the group does not require translations. To extract the linear
    part of the basis: lin = basis[:-1, :-1].

    This function focuses on 'classic' Lie groups. Note that, while it
    is commonly used in registration software, we do not have a
    "9-parameter affine" (translations + rotations + zooms),
    because such transforms do not form a group; that is, their inverse
    may contain shears.

    Parameters
    ----------
    group : {'T', 'SO', 'SE', 'D', 'CSO', 'SL', 'GL+', 'Aff+'}, default='SE'
        Group that should be encoded by the basis set:
            * 'T'    or 'translation' : Translations
            * 'SO'   or 'rotation'    : Special Orthogonal (rotations)
            * 'SE'   or 'rigid'       : Special Euclidean (translations + rotations)
            * 'D'    or 'dilation'    : Dilations (translations + isotropic scalings)
            * 'CSO'  or 'similitude'  : Conformal Special Orthogonal
                                        (translations + rotations + isotropic scalings)
            * 'SL'                    : Special Linear (rotations + isovolumic zooms + shears)
            * 'GL+'  or 'linear'      : General Linear [det>0] (rotations + zooms + shears)
            * 'Aff+' or 'affine'      : Affine [det>0] (translations + rotations + zooms + shears)
    dim : {1, 2, 3}, default=3
        Dimension
    dtype : torch.dtype, optional
        Data type of the returned array

    Returns
    -------
    basis : (F, dim+1, dim+1) tensor[dtype]
        Basis set, where ``F`` is the number of basis functions.

    """
    # TODO:
    # - other groups?

    # Authors
    # -------
    # .. John Ashburner <j.ashburner@ucl.ac.uk> : original Matlab code
    # .. Yael Balbastre <yael.balbastre@gmail.com> : Python code

    if group not in affine_basis_choices:
        for key, aliases in affine_basis_aliases.items():
            if group in aliases:
                group = key
                break
    if group not in affine_basis_choices:
        raise ValueError('group must be one of {}.'
                         .format(affine_basis_choices))

    subbases = [affine_subbasis(sub, dim, dtype=dtype)
                for sub in affine_basis_components[group]]
    return subbases[0] if len(subbases) == 1 else np.concatenate(subbases)


def affine_basis_size(group, dim=3):
    """Return the number of parameters in a given group."""

    if group not in affine_basis_choices:
        for key, aliases in affine_basis_aliases.items():
            if group in aliases:
                group = key
                break
    if group not in affine_basis_choices:
        raise ValueError('group must be one of {}.'
                         .format(affine_basis_choices))
    if group == 'T':
        return dim
    elif group == 'SO':
        return dim * (dim - 1) // 2
    elif group == 'SE':
        return dim + dim * (dim - 1) // 2
    elif group == 'D':
        return dim + 1
    elif group == 'CSO':
        return dim + dim * (dim - 1) // 2 + 1
    elif group == 'SL':
        return (dim + 1) * (dim - 1)
    elif group == 'GL+':
        return dim * dim
    elif group == 'Aff+':
        return (dim + 1) * dim


def affine_subbasis_size(basis, dim=3):
    """Return the number of parameters in a given group."""

    if basis not in affine_subbasis_choices:
        for key, aliases in affine_subbasis_aliases.items():
            if basis in aliases:
                basis = key
                break
    if basis not in affine_subbasis_choices:
        raise ValueError('basis must be one of {}.'
                         .format(affine_subbasis_choices))
    if basis in ('T', 'Z'):
        return dim
    elif basis in ('S', 'SC', 'R'):
        return dim * (dim - 1) // 2
    elif basis == 'Z0':
        return dim - 1
    elif basis == 'I':
        return 1


def build_affine_basis(*basis, dim=None, dtype=None):
    """Transform Affine Lie bases into tensors.

    Signatures
    ----------
    basis = build_affine_basis(basis)
    basis1, basis2, ... = build_affine_basis(basis1, basis2, ...)

    Parameters
    ----------
    *basis : basis_like or sequence[basis_like]
        A basis_like is a str or ([F], D+1, D+1) tensor_like that
        describes a basis. If several arguments are provided, each one
        is built independently.
    dim : int, optional
        Dimensionality. If None, infer.
    dtype : torch.dtype, optional
        Output data type

    Returns
    -------
    *basis : (F, D+1, D+1) tensor
        Basis sets.

    """
    if basis and (isinstance(basis[-1], int) or basis[-1] is None):
        *basis, dim = basis
    opts = dict(dtype=dtype, dim=dim)
    bases = [_build_affine_basis(b, **opts) for b in basis]
    return bases[0] if len(bases) == 1 else tuple(bases)


def _build_affine_basis(basis, dim=None, dtype=None):
    """Actual implementation: only one basis set."""

    # Helper to convert named bases to matrices
    def name_to_basis(name, dim, dtype):
        basename = name.split('[')[0]
        if basename in affine_subbasis_choices:
            return affine_subbasis(name, dim, dtype=dtype)
        elif basename in affine_basis_choices:
            return affine_basis(name, dim, dtype=dtype)
        else:
            raise ValueError('Unknown basis name {}.'.format(basename))

    # make list
    basis = ensure_list(basis)
    built_bases = [b for b in basis if not isinstance(b, str)]

    # check dimension
    dims = [dim] if dim else []
    dims = dims + [b.shape[-1] - 1 for b in built_bases]
    dims = set(dims)
    if not dims:
        dim = 3
    elif len(dims) > 1:
        raise ValueError('Dimension not consistent across bases.')
    else:
        dim = dims.pop()

    # build bases
    basis0 = basis
    basis = []
    for b in basis0:
        if isinstance(b, str):
            b = name_to_basis(b, dim, dtype)
        else:
            b = np.asanyarray(b)
        while b.ndim < 3:
            b = b[None]
        if b.shape[-2] != b.shape[-1] or b.ndim != 3:
            raise ValueError('Expected basis with shape (B, D+1, D+1) '
                             'but got {}'.format(tuple(b.shape)))
        basis.append(b)
    basis = np.concatenate(basis, axis=0)

    return basis
