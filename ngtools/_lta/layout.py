# stdlib
import itertools

# externals
import numpy as np

# internals
from .utils import ensure_list


def volume_axis(*args, **kwargs):
    """Describe an axis of a volume of voxels.

    Signature
    ---------
    * ``def volume_axis(index, flipped=False)``
    * ``def volume_axis(name)``
    * ``def volume_axis(axis)``

    Parameters
    ----------
    index : () array_like[int]
        Index of the axis in 'direct' space (RAS)

    flipped : () array_like[bool], default=False
        Whether the axis is flipped or not.

    name : {'R' or 'L', 'A' or 'P', 'S' or 'I'}
        Name of the axis, according to the neuroimaging convention:
        * 'R' for *left to Right* (index=0, flipped=False) or
          'L' for *right to Left* (index=0, flipped=True)
        * 'A' for *posterior to Anterior* (index=1, flipped=False) or
          'P' for *anterior to Posterior* (index=1, flipped=True)
        * 'S' for *inferior to Superior* (index=2, flipped=False) or
          'I' for *superior to Inferior* (index=2, flipped=True)

    axis : (2,) array_like[int]
        ax[0] = index
        ax[1] = flipped
        Description of the axis.

    Returns
    -------
    axis : (2,) tensor[int]
        Description of the axis.
        * `ax[0] = index`
        * `ax[1] = flipped`

    """
    def axis_from_name(name):
        name = name.upper()
        if name == 'R':
            return np.asarray([0, 0], dtype="int")
        elif name == 'L':
            return np.asarray([0, 1], dtype="int")
        elif name == 'A':
            return np.asarray([1, 0], dtype="int")
        elif name == 'P':
            return np.asarray([1, 1], dtype="int")
        elif name == 'S':
            return np.asarray([2, 0], dtype="int")
        elif name == 'I':
            return np.asarray([2, 1], dtype="int")

    def axis_from_index(index, flipped=False):
        index = np.asarray(index).reshape([])
        flipped = np.asarray(flipped).reshape([])
        return np.asarray([index, flipped], dtype="int")

    def axis_from_axis(ax):
        ax = np.asarray(ax, dtype="int").flatten()
        if ax.size != 2:
            raise ValueError('An axis should have two elements. Got {}.'
                             .format(ax.size))
        return ax

    # Dispatch based on input types or keyword arguments
    args = list(args)
    if len(args) > 0:
        if isinstance(args[0], str):
            return axis_from_name(*args, **kwargs)
        else:
            args[0] = np.asarray(args[0])
            if args[0].numel() == 1:
                return axis_from_index(*args, **kwargs)
            else:
                return axis_from_axis(*args, **kwargs)
    else:
        if 'name' in kwargs.keys():
            return axis_from_name(*args, **kwargs)
        elif 'index' in kwargs.keys():
            return axis_from_index(*args, **kwargs)
        else:
            return axis_from_axis(*args, **kwargs)


# Mapping from (index, flipped) to axis name
_axis_names = [['R', 'L'], ['A', 'P'], ['S', 'I']]


def volume_axis_to_name(axis):
    """Return the (neuroimaging) name of an axis. Its index must be < 3.

    Parameters
    ----------
    axis : (2,) array_like

    Returns
    -------
    name : str

    """
    index, flipped = axis
    if index >= 3:
        raise ValueError('Index names are only defined up to dimension 3. '
                         'Got {}.'.format(index))
    return _axis_names[index][flipped]


def volume_layout(*args, **kwargs):
    """Describe the layout of a volume of voxels.

    A layout is characterized by a list of axes. See `volume_axis`.

    Signature
    ---------
    volume_layout(ndim=3)
    volume_layout(name)
    volume_layout(axes)
    volume_layout(index, flipped=False)

    Parameters
    ----------
    ndim : int, default=3
        Dimension of the space.
        This version of the function always returns a directed layout
        (identity permutation, no flips), which is equivalent to 'RAS'
        but in any dimension.

    name : str
        Permutation of axis names, according to the neuroimaging convention:
        * 'R' for *left to Right* (index=0, flipped=False) or
          'L' for *right to Left* (index=0, flipped=True)
        * 'A' for *posterior to Anterior* (index=1, flipped=False) or
          'P' for *anterior to Posterior* (index=1, flipped=True)
        * 'S' for *inferior to Superior* (index=2, flipped=False) or
          'I' for *superior to Inferior* (index=2, flipped=True)
        The number of letters defines the dimension of the matrix
        (`ndim = len(name)`).

    axes : (ndim, 2) array_like[int]
        List of objects returned by `axis`.

    index : (ndim, ) array_like[int]
        Index of the axes in 'direct' space (RAS)

    flipped : (ndim, ) array_like[bool], default=False
        Whether each axis is flipped or not.

    Returns
    -------
    layout : (ndim, 2) tensor[int]
        Description of the layout.

    """
    def layout_from_dim(dim):
        return volume_layout(list(range(dim)), flipped=False)

    def layout_from_name(name):
        return volume_layout([volume_axis(a) for a in name])

    def layout_from_index(index, flipped=False):
        index = np.asarray(index, dtype="int").flatten()
        ndim = index.shape[0]
        flipped = np.asarray(flipped, dtype="int").flatten()
        if flipped.shape[0] == 1:
            flipped = np.repeat(flipped, ndim, axis=0)
        return np.stack((index, flipped), axis=-1)

    def layout_from_axes(axes):
        axes = np.asarray(axes, dtype="int")
        if axes.ndim != 2 or axes.shape[1] != 2:
            raise ValueError('A layout should have shape (ndim, 2). Got {}.'
                             .format(axes.shape))
        return axes

    # Dispatch based on input types or keyword arguments
    args = list(args)
    if len(args) > 0:
        if isinstance(args[0], str):
            layout = layout_from_name(*args, **kwargs)
        else:
            args[0] = np.asarray(args[0])
            if args[0].ndim == 0:
                layout = layout_from_dim(*args, **kwargs)
            elif args[0].ndim == 2:
                layout = layout_from_axes(*args, **kwargs)
            else:
                layout = layout_from_index(*args, **kwargs)
    else:
        if 'dim' or 'ndim' in kwargs.keys():
            layout = layout_from_dim(*args, **kwargs)
        elif 'name' in kwargs.keys():
            layout = layout_from_name(*args, **kwargs)
        elif 'index' in kwargs.keys():
            layout = layout_from_index(*args, **kwargs)
        else:
            layout = layout_from_axes(*args, **kwargs)

    # Remap axes indices if not contiguous
    backend = dict(dtype=layout.dtype)
    axes = layout[:, 0]
    remap = np.argsort(axes)
    axes[remap] = np.arange(len(axes), **backend)
    layout = np.stack((axes, layout[:, 1]), axis=-1)
    return layout


def volume_layout_to_name(layout):
    """Return the (neuroimaging) name of a layout.

    Its length must be <= 3 (else, we just return the permutation and
    flips), e.g. '[2, 3, -1, 4]'

    Parameters
    ----------
    layout : (dim, 2) array_like

    Returns
    -------
    name : str

    """
    layout = volume_layout(layout)
    if len(layout) > 3:
        layout = [('-' if bool(f) else '') + str(int(p)) for p, f in layout]
        return '[' + ', '.join(layout) + ']'
    names = [volume_axis_to_name(axis) for axis in layout]
    return ''.join(names)


def iter_layouts(ndim):
    """Compute all possible layouts for a given dimensionality.

    Parameters
    ----------
    ndim : () array_like
        Dimensionality (rank) of the space.

    Returns
    -------
    layouts : (nflip*nperm, ndim, 2) tensor[int]
        All possible layouts.
        * nflip = 2 ** ndim     -> number of flips
        * nperm = ndim!         -> number of permutations

    """

    # First, compute all possible directed layouts on one hand,
    # and all possible flips on the other hand.
    axes = np.arange(ndim, dtype="int")
    layouts = np.asarray(itertools.permutations(axes))          # [P, D]
    flips = np.asarray(itertools.product([0, 1], repeat=ndim))  # [F, D]

    # Now, compute combination (= cartesian product) of both
    # We replicate each tensor so that shapes match and stack them.
    nb_layouts = len(layouts)
    nb_flips = len(flips)
    layouts = layouts[None, ...]
    layouts = np.repeat(layouts, nb_flips, axis=0)  # [F, P, D]
    flips = flips[:, None, :]
    flips = np.repeat(flips, nb_layouts, axis=1)    # [F, P, D]
    layouts = np.stack([layouts, flips], axis=-1)

    # Finally, flatten across repeats
    layouts = layouts.reshape([-1, ndim, 2])    # [F*P, D, 2]

    return layouts


def layout_matrix(layout, voxel_size=1., shape=0., dtype=None):
    """Compute the origin affine matrix for different voxel layouts.

    Resources
    ---------
    .. https://nipy.org/nibabel/image_orientation.html
    .. https://nipy.org/nibabel/neuro_radio_conventions.html

    Parameters
    ----------
    layout : str or (ndim, 2) array_like[int]
        See `affine.layout`

    voxel_size : (ndim,) array_like, default=1
        Voxel size of the lattice

    shape : (ndim,) array_like, default=0
        Shape of the lattice

    dtype : torch.dtype, optional
        Data type of the matrix

    Returns
    -------
    mat : (ndim+1, ndim+1) tensor[dtype]
        Corresponding affine matrix.

    """

    # Author
    # ------
    # .. Yael Balbastre <yael.balbastre@gmail.com>

    # Extract info from layout
    layout = volume_layout(layout)
    dim = len(layout)
    perm = invert_permutation(layout[:, 0])
    flip = layout[:, 1].astype("bool")

    # ensure tensor
    voxel_size = np.asarray(voxel_size)
    shape = np.asarray(shape)

    # ensure dim
    shape = np.asarray(ensure_list(shape.tolist(), dim))
    voxel_size = np.asarray(ensure_list(voxel_size.tolist(), dim))
    zero = np.zeros(dim, dtype=dtype)

    # Create matrix
    mat = np.diag(voxel_size)                               # [D, D]
    mat = mat[perm, :]                                      # [D, D]

    mflip = np.ones(dim)                                    # [D]
    mflip = np.where(flip, -mflip, mflip)                   # [D]
    mflip = np.diag(mflip)                                  # [D, D]
    shift = np.where(flip, shape[perm], zero)               # [D]
    mflip = np.concatenate((mflip, shift[:, None]), axis=1)         # [D, D+1]

    mat = np.matmul(mat, mflip)                             # [D, D+1]
    mat = np.concatenate([mat, [[0] * dim + [1]]], axis=0)            # [D+1, D+1]
    return mat


def invert_permutation(perm):
    """Return the inverse of a permutation

    Parameters
    ----------
    perm : (..., N) array_like
        Permutations. A permutation is a shuffled set of indices.

    Returns
    -------
    iperm : (..., N) array
        Inverse permutation.

    """
    perm = np.asarray(perm)
    shape = perm.shape
    perm = perm.reshape([-1, shape[-1]])
    n = perm.shape[-1]
    k = perm.shape[0]
    identity = np.arange(n, dtype="int")[None, ...]
    identity = np.broadcast_to(identity, [k, n])  # Repeat without allocation
    iperm = np.empty_like(perm)
    np.put_along_axis(iperm, perm, identity, -1)
    iperm = iperm.reshape(shape)
    return iperm
