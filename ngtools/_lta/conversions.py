"""Tools to convert between affine representations."""

# externals
import numpy as np

# internals
from .layout import layout_matrix
from .lie import lie_to_matrix, matrix_to_lie


class Affine:
    """Base class for all affine transformations"""

    def xras(self):
        """First column of a compact Vox2World matrix (MGH convention)"""
        return self.xyzcras()[..., 0]

    def yras(self):
        """Second column of a compact Vox2World matrix (MGH convention)"""
        return self.xyzcras()[..., 1]

    def zras(self):
        """Third column of a compact Vox2World matrix (MGH convention)"""
        return self.xyzcras()[..., 2]

    def cras(self):
        """Last column (translation) of a compact Vox2World matrix
         (MGH convention)"""
        return self.xyzcras()[..., -1]

    def xyzcras(self):
        """Compact form of a Vox2World matrix (MGH convention)"""
        return self.compact()

    def sform(self):
        """Alias for the compact matrix form of a Vox2World (NIfTI convention)"""
        return self.homogenous()[..., :-1, :]

    def compact(self):
        """Compact form of a homogeneous affine matrix (last row dropped)"""
        return self.homogenous()[..., :-1, :]

    def homogenous(self):
        """Homogeneous matrix form."""
        matrix = self.compact()
        new_shape = list(matrix.shape)
        new_shape[-2] += 1
        new_matrix = np.zeros_like(matrix, shape=new_shape)
        new_matrix[..., :-1, :] = matrix
        new_matrix[..., -1, -1] = 1
        return new_matrix

    def affine(self):
        """Alias for the homogeneous matrix form (NiBabel convention)"""
        return self.homogenous()

    def translation(self):
        """Translation component"""
        return self.compact()[..., -1]

    def linear(self):
        """Linear component"""
        return self.compact()[..., :-1]

    def rotation(self):
        """Rotation component (in the least squares sense)"""
        basis = 'GL+'
        matrix = self.linear().homogeneous()
        lie = matrix_to_lie(matrix, basis)
        rot = lie[:3]
        return lie_to_matrix(rot, basis, ndim=3)

    def quaternion(self):
        """Rotation component encoded with quaternion"""
        return RotationMatrix(self.rotation).quaternion()

    def qform(self):
        """Rotation component encoded with quaternion (NIfTI convention)"""
        return self.quaternion()

    def axisangle(self):
        """Rotation encoded by an axis of rotation and an angle"""
        return Quaternion(self.quaternion).axisangle()

    def axis(self):
        """Rotation encoded by an axis of rotation and an angle"""
        return Quaternion(self.quaternion).axis()

    def angle(self):
        """Rotation encoded by an axis of rotation and an angle"""
        return Quaternion(self.quaternion).angle()

    def voxel_size(self):
        """Voxel size of a Voxel2World"""
        return self.linear().square().sum(-1).sqrt()


class HomogeneousAffineMatrix(Affine):
    """ (D+1)x(D+1) homogeneous affine matrix"""

    def __init__(self, matrix):
        matrix = np.asanyarray(matrix)
        if matrix.shape[-1] != matrix.shape[-2]:
            raise ValueError('Expected square matrix')
        self.matrix = matrix

    def homogenous(self):
        return self.matrix


class CompactAffineMatrix(Affine):
    """ Dx(D+1) compact affine matrix"""

    def __init__(self, matrix):
        matrix = np.asanyarray(matrix)
        if matrix.shape[-1] - 1 != matrix.shape[-2]:
            raise ValueError('Expected compact matrix')
        self.matrix = matrix

    def compact(self):
        return self.matrix


class XYZC(Affine):
    """Affine matrix stored using independent columns"""

    def __init__(self, x, y, z, c):
        self.x = x
        self.y = y
        self.z = z
        self.c = c

    def xras(self):
        return self.x

    def yras(self):
        return self.y

    def zras(self):
        return self.z

    def cras(self):
        return self.c

    def compact(self):
        return np.stack([self.x, self.y, self.z, self.c], axis=-1)


class Linear(Affine):
    """Affine subgroup (no translation)"""

    def compact(self):
        matrix = self.linear()
        new_shape = list(matrix.shape)
        new_shape[-1] += 1
        new_matrix = np.zeros_like(matrix, shape=new_shape)
        new_matrix[..., :-1] = matrix
        return new_matrix


class Rotation(Linear):
    """Linear subgroup (orthogonal)"""

    def linear(self):
        return self.rotation()


class RotationMatrix(Rotation):
    """Rotation encoded using a DxD orthogonal matrix."""

    def __init__(self, matrix):
        matrix = np.asanyarray(matrix)
        if matrix.shape[-1] != matrix.shape[-2]:
            raise ValueError('Expected a square matrix')
        self.matrix = matrix

    def rotation(self):
        return self.matrix

    def quaternion(self):
        def vpos(mat, trace):
            r = trace.add(1).sqrt()
            s = r.reciprocal().div(2)
            i = (mat[..., 2, 1] - mat[..., 1, 2]) * s
            j = (mat[..., 0, 2] - mat[..., 2, 0]) * s
            k = (mat[..., 1, 0] - mat[..., 0, 1]) * s
            w = r.div(2)
            return np.stack([i, j, k, w], -1)

        def vneg(mat):
            r = (1 + mat[..., 0, 0] - mat[..., 1, 1] - mat[..., 2, 2]).sqrt()
            s = r.reciprocal().div(2)
            w = (mat[..., 2, 1] - mat[..., 1, 2]) * s
            i = r.div(2)
            j = (mat[..., 0, 1] - mat[..., 1, 0]) * s
            k = (mat[..., 2, 0] - mat[..., 0, 2]) * s
            return np.stack([i, j, k, w], -1)

        trace = self.matrix.trace()
        mask = trace > 0
        return np.where(mask, vpos(self.matrix, trace), vneg(self.matrix))


class Quaternion(Rotation):
    """Rotation stored in quanternion form"""

    def __init__(self, *args):
        """

        Parameters
        ----------
        Either
            quat : (..., 4) tensor
        Or
            orientation : (..., 3) tensor
            attitude : (...) tensor
        Or
            i, j, k, r : tensors

        """
        if len(args) == 1:
            ijkr = np.asanyarray(args[0])
            i = ijkr[..., 0]
            j = ijkr[..., 1]
            k = ijkr[..., 2]
            r = ijkr[..., 3]
        elif len(args) == 2:
            ijk, r = args
            i = ijk[..., 0]
            j = ijk[..., 1]
            k = ijk[..., 2]
        elif len(args) == 4:
            i, j, k, r = args
        else:
            raise ValueError('Expected 1, 2 or 4 arguments')
        self.i = i
        self.j = j
        self.k = k
        self.r = r

    def quaternion(self):
        return self.i, self.j, self.k, self.r

    def rotation(self):
        i = self.i
        j = self.j
        k = self.k
        r = self.r
        matrix = [[1-2*(j**2 + k**2), 2*(i*j - k*r), 2*(i*k + j*r)],
                  [2*(i*j + k*r), 1 - 2*(i**2 + k**2), 2*(j*k - i*r)],
                  [2*(i*k - j*r), 2*(j*k + i*r), 1 - 2*(i**2 + j**2)]]
        matrix = np.asanyarray(matrix)
        matrix = np.moveaxis(matrix, [0, 1], [-2, -1])
        return matrix

    def axisangle(self):
        axis = np.stack([self.i, self.j, self.k], axis=-1)
        nrm = (axis*axis).sum(-1, keepdims=True).sqrt()
        axis = axis / nrm
        nrm = nrm[..., 0]
        angle = 2*np.atan2(nrm, self.r)[..., None]
        return np.concatenate([axis, angle], axis=-1)

    def axis(self):
        axis = np.stack([self.i, self.j, self.k], axis=-1)
        axis = axis / (axis*axis).sum(-1, keepdims=True).sqrt()
        return axis

    def angle(self):
        axis = np.stack([self.i, self.j, self.k], axis=-1)
        axis = (axis*axis).sum(-1).sqrt()
        angle = 2*np.atan2(axis, self.r)
        return angle


class AxisAngle(Rotation):
    """Rotation stored using an axis of rotation and an angle."""

    def __init__(self, axis, angle):
        self.ax = axis
        self.theta = angle

    def quaternion(self):
        return Quaternion(self.ax * self.theta.div(2).sin(),
                          self.theta.div(2).cos()).quaternion()

    def axis(self):
        return self.ax

    def angle(self):
        return self.theta

    def axisangle(self):
        return np.concatenate([self.axis, self.angle[..., None]], -1)


class Orientation(Affine):
    """Orientation matrix that contains only an anisotropic scaling,
    flips and permutations, and a translation.

    This is the simplest possible voxel-to-world matrix for
    images of real-world objects.
    """

    def __init__(self, shift, scale, orientation='RAS'):
        self.shift = shift
        self.scale = scale
        self.orientation = orientation

    def affine(self):
        aff = layout_matrix(self.orientation, self.scale)
        aff[:-1, -1] = self.shift
        return aff
