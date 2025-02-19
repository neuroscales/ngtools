# stdlib
import re
from warnings import warn

# externals
import numpy as np

# internals
from .conversions import XYZC, HomogeneousAffineMatrix, Orientation
from .layout import layout_matrix
from .utils import ensure_list


def ensure_homogeneous(mat):
    if mat.shape[-2] != mat.shape[-1]:
        new_shape = list(mat.shape)
        new_shape[-2] += 1
        new_mat = np.zeros_like(mat, shape=new_shape)
        new_mat[..., :-1, :] = mat
        new_mat[..., -1, -1] = 1
        return new_mat
    return mat


def get_voxel_size(mat):
    ndim = mat.shape[-1] - 1
    mat = mat[..., :ndim, :ndim]
    return np.sqrt(np.sum(mat*mat, axis=-2))


def make_vector(x, *args, **kwargs):
    return np.asarray(ensure_list(np.asarray(x).tolist(), *args, **kwargs))


def affine_matmul(a, b):
    """Matrix-matrix product of affine matrices.

    Parameters
    ----------
    a : (..., ndim+1, ndim+1) tensor
        Affine matrix
    b : (..., ndim+1, ndim+1) tensor
        Affine matrix

    Returns
    -------
    affine_times_matrix : (..., ndim+1, ndim+1) tensor

    """
    Za = a[..., :-1, :-1]
    Ta = a[..., :-1, -1:]
    Zb = b[..., :-1, :-1]
    Tb = b[..., :-1, -1:]
    Z = np.matmul(Za, Zb)
    T = np.matmul(Za, Tb) + Ta
    out = np.concatenate((Z, T), axis=-1)
    out = ensure_homogeneous(out)
    return out


def affine_inv(affine):
    """Inverse of an affine matrix.

    If the input matrix is not symmetric with respect to its input
    and output spaces, a pseudo-inverse is returned instead.

    Parameters
    ----------
    affine : (..., ndim[+1], ndim+1) array
        Input affine

    Returns
    -------
    inv_affine : (..., ndim[+1], ndim+1) array

    """
    homogeneous = True
    ndim = affine.shape[-1] - 1
    if affine.shape[-2] != ndim + 1:
        homogeneous = False
        affine = ensure_homogeneous(affine)
    affine = np.linalg.inv(affine)
    if not homogeneous:
        affine = affine[..., :-1, :]
    return affine


# Regex patterns for different value types
patterns = {
    int: r'(\d+)',
    float: r'([\+\-]?\d+\.?\d*(?:[eE][\+\-]?\d+)?)',
    str: r'(.*)\s*$'
}


def read_key(line, key_dict=None):
    """Read one `key = value` line from an LTA file

    Parameters
    ----------
    line : str
    format : type in {int, float, str}

    Returns
    -------
    object or tuple or None

    """
    key_dict = key_dict or dict()

    line = line.split('\r\n')[0]  # remove eol (windows)
    line = line.split('\n')[0]    # remove eol (unix)
    line = line.split('#')[0]     # remove hanging comments
    pattern = r'^\s*(\S+)\s*=\s*(.*)$'
    match = re.match(pattern, line)
    if not match:
        warn(f'cannot read line: "{line}"', RuntimeWarning)
        return None, None

    key = match.group(1)
    value = match.group(2).rstrip()
    if key in key_dict:
        format = key_dict[key]
        if isinstance(format, type):
            pattern = patterns[format]
        else:
            pattern = r'\s*'.join([patterns[fmt] for fmt in format])
        match = re.match(pattern, value)
        if match:
            if isinstance(format, type):
                value = format(match.group(1))
            else:
                value = tuple(fmt(v) for v, fmt in zip(match.groups(), format))
        else:
            warn(f'cannot parse value: "{value}"', RuntimeWarning)
    return key, value


def read_values(line, format):
    """Read one `*values` line from an LTA file

    Parameters
    ----------
    line : str
    format : [sequence of] type
        One of {int, float, str}

    Returns
    -------
    object or tuple or None

    """
    line = line.split('\r\n')[0]  # remove eol (windows)
    line = line.split('\n')[0]    # remove eol (unix)
    line = line.split('#')[0]     # remove hanging comments
    pattern = r'\s*'
    if isinstance(format, type):
        reformat = [patterns[format]]
    else:
        reformat = [patterns[fmt] for fmt in format]
    pattern += r'\s*'.join(reformat)
    value = re.match(pattern, line)
    if value:
        if isinstance(format, type):
            value = format(value.group(1))
        else:
            value = tuple(fmt(v) for v, fmt in zip(value.groups(), format))
    else:
        warn(f'cannot read line: "{line}"', RuntimeWarning)
    return value


def write_key(key, value):
    """Write a `key = value` line in an LTA file.

    Parameters
    ----------
    key : str
    value : int or float or str

    Returns
    -------
    str

    """
    if isinstance(value, (str, int, float)):
        return f'{key:9s} = {value}'
    else:
        return f'{key:9s} = ' + ' '.join([str(v) for v in value])


def write_values(value):
    """Write a `*values` line in an LTA file.

    Parameters
    ----------
    value : [sequence of] int or float or str

    Returns
    -------
    str

    """
    if isinstance(value, (str, int, float)):
        return str(value)
    else:
        return ' '.join([str(v) for v in value])


def fs_to_affine(shape, voxel_size=1., x=None, y=None, z=None, c=0.,
                 source='voxel', dest='ras'):
    """Transform FreeSurfer orientation parameters into an affine matrix.

    The returned matrix is effectively a "<source> to <dest>" transform.

    Parameters
    ----------
    shape : sequence of int
    voxel_size : [sequence of] float, default=1
    x : [sequence of] float, default=[1, 0, 0]
    y: [sequence of] float, default=[0, 1, 0]
    z: [sequence of] float, default=[0, 0, 1]
    c: [sequence of] float, default=0
    source : {'voxel', 'physical', 'ras'}, default='voxel'
    dest : {'voxel', 'physical', 'ras'}, default='ras'

    Returns
    -------
    affine : (4, 4) array

    """
    shape = make_vector(shape)
    voxel_size = make_vector(voxel_size)
    dim = len(list(shape))
    if x is None:
        x = [1, 0, 0]
    if y is None:
        y = [0, 1, 0]
    if z is None:
        z = [0, 0, 1]
    x = make_vector(x, dim)
    y = make_vector(y, dim)
    z = make_vector(z, dim)
    c = make_vector(c, dim)

    shift = shape / 2.
    shift = -shift * voxel_size
    vox2phys = Orientation(shift, voxel_size).affine()
    phys2ras = XYZC(x, y, z, c).affine()

    affines = []
    if source.lower().startswith('vox'):
        affines.append(vox2phys)
        middle_space = 'phys'
    elif source.lower().startswith('phys'):
        if dest.lower().startswith('vox'):
            affines.append(affine_inv(vox2phys))
            middle_space = 'vox'
        else:
            affines.append(phys2ras)
            middle_space = 'ras'
    elif source.lower() == 'ras':
        affines.append(affine_inv(phys2ras))
        middle_space = 'phys'
    else:
        # We need a matrix to switch orientations
        affines.append(layout_matrix(source))
        middle_space = 'ras'

    if dest.lower().startswith('phys'):
        if middle_space == 'vox':
            affines.append(vox2phys)
        elif middle_space == 'ras':
            affines.append(affine_inv(phys2ras))
    elif dest.lower().startswith('vox'):
        if middle_space == 'phys':
            affines.append(affine_inv(vox2phys))
        elif middle_space == 'ras':
            affines.append(affine_inv(phys2ras))
            affines.append(affine_inv(vox2phys))
    elif dest.lower().startswith('ras'):
        if middle_space == 'phys':
            affines.append(phys2ras)
        elif middle_space.lower().startswith('vox'):
            affines.append(vox2phys)
            affines.append(phys2ras)
    else:
        if middle_space == 'phys':
            affines.append(affine_inv(phys2ras))
        elif middle_space == 'vox':
            affines.append(vox2phys)
            affines.append(phys2ras)
        layout = layout_matrix(dest)
        affines.append(affine_inv(layout))

    affine, *affines = affines
    for aff in affines:
        affine = affine_matmul(aff, affine)
    return affine


def affine_to_fs(affine, shape, source='voxel', dest='ras'):
    """Convert an affine matrix into FS parameters (vx/cosine/shift)

    Parameters
    ----------
    affine : (4, 4) array
    shape : (int, int, int)
    source : {'voxel', 'physical', 'ras'}, default='voxel'
    dest : {'voxel', 'physical', 'ras'}, default='ras'

    Returns
    -------
    voxel_size : (float, float, float)
    x : (float, float, float)
    y : (float, float, float)
    z: (float, float, float)
    c : (float, float, float)

    """

    affine = np.asarray(affine)
    vx = get_voxel_size(affine)
    shape = np.asarray(shape)
    source = source.lower()[0]
    dest = dest.lower()[0]

    shift = shape / 2.
    shift = -shift * vx
    vox2phys = Orientation(shift, vx).affine()

    if (source, dest) in (('v', 'p'), ('p', 'v')):
        phys2ras = np.eye(4)

    elif (source, dest) in (('v', 'r'), ('r', 'v')):
        if source == 'r':
            affine = affine_inv(affine)
        phys2vox = affine_inv(vox2phys)
        phys2ras = affine_matmul(affine, phys2vox)

    else:
        assert (source, dest) in (('p', 'r'), ('r', 'p'))
        if source == 'r':
            affine = affine_inv(affine)
        phys2ras = affine

    phys2ras = HomogeneousAffineMatrix(phys2ras)
    return (vx.tolist(), phys2ras.xras().tolist(), phys2ras.yras().tolist(),
            phys2ras.zras().tolist(), phys2ras.cras().tolist())
