# stdlib
from warnings import warn

# externals
import numpy as np

# internals
from .basis import build_affine_basis
from .expm import expm
from .utils import ensure_list


def multi_dot(x):
    if len(x) > 1:
        return np.linalg.multi_dot(x)
    else:
        return x


def lie_to_matrix(prm, *basis, ndim=None):
    r"""Reconstruct an affine matrix from its Lie parameters.

    Affine matrices are encoded as product of sub-matrices, where
    each sub-matrix is encoded in a Lie algebra.
    ..math:: M   = exp(A_1) \times ... \times exp(A_n)
    ..math:: A_i = \sum_k = p_{ik} B_{ik}

    Examples
    --------
    ```python
    >> prm = torch.randn(6)
    >> # from a classic Lie group
    >> A = affine_matrix_lie(prm, 'SE', ndim=3)
    >> # from a user-defined Lie group
    >> A = affine_matrix_lie(prm, ['Z', 'R[0]', 'T[1]', 'T[2]'], ndim=3)
    >> # from two Lie groups
    >> A = affine_matrix_lie(prm, 'Z', 'R', ndim=3)
    >> B = affine_matrix_lie(prm[:3], 'Z', ndim=3)
    >> C = affine_matrix_lie(prm[3:], 'R', ndim=3)
    >> assert torch.allclose(A, B @ C)
    >> # from a pre-built basis
    >> basis = affine_basis('SE', ndim=3)
    >> A = affine_matrix_lie(prm, basis, ndim=3)
    ```

    Parameters
    ----------
    prm : (..., nb_basis)
        Parameters in the Lie algebra(s).

    *basis : basis_like, default='CSO'
        A basis_like is a (sequence of) (F, D+1, D+1) tensor or string.
        The number of parameters (for each batch element) should be equal
        to the total number of bases (the sum of all bases across sub-bases).

    ndim : int, default=guess or 3
        If not provided, the function tries to guess it from the shape
        of the basis matrices. If the dimension cannot be guessed
        (because all bases are named bases), the default is 3.

    Returns
    -------
    mat : (..., ndim+1, ndim+1) tensor
        Reconstructed affine matrix.

    """

    # Author
    # ------
    # .. Yael Balbastre <yael.balbastre@gmail.com>

    # Input parameters
    prm = np.asarray(prm)
    info = dict(dtype=prm.dtype)

    # Make sure basis is a vector_like of (F, D+1, D+1) tensor_like
    if len(basis) == 0:
        basis = ['CSO']
    if basis and (isinstance(basis[-1], int) or basis[-1] is None):
        *basis, ndim = basis
    basis = build_affine_basis(*basis, ndim, **info)
    basis = ensure_list(basis)

    # Check length
    nb_basis = sum([len(b) for b in basis])
    if prm.shape[-1] != nb_basis:
        raise ValueError('Number of parameters and number of bases do '
                         'not match. Got {} and {}'
                         .format(len(prm), nb_basis))

    # Reconstruct each sub matrix
    n_prm = 0
    mats = []
    for a_basis in basis:
        nb_prm = len(a_basis)
        a_prm = prm[..., n_prm:(n_prm+nb_prm)]
        mats.append(expm(a_prm, a_basis))
        n_prm += nb_prm

    # Matrix product
    if len(mats) > 1:
        affine = multi_dot(mats)
    else:
        affine = mats[0]
    return affine


def classic_to_matrix(prm=None, ndim=3, *,
                      translations=None,
                      rotations=None,
                      zooms=None,
                      shears=None):
    """Build an affine matrix in the "classic" way (no Lie algebra).

    Parameters can either be provided already concatenated in the last
    dimension (`prm=...`) or as individual components (`translations=...`)

    Parameters
    ----------
    prm : (..., K) tensor_like
        Affine parameters, ordered -- in the last dimension -- as
        `[*translations, *rotations, *zooms, *shears]`
        Rotation parameters should be expressed in radians.
    ndim : () tensor_like[int]
        Dimensionality.

    Alternative Parameters
    ----------------------
    translations : (..., ndim) tensor_like, optional
        Translation parameters (along X, Y, Z).
    rotations : (..., ndim*(ndim-1)//2) tensor_like, optional
        Rotation parameters, in radians (about X, Y, Z)
    zooms : (..., ndim|1) tensor_like, optional
        Zoom parameters (along X, Y, Z).
    shears : (..., ndim*(ndim-1)//2) tensor_like, optional
        Shear parameters (about XY, XZ, YZ).


    Returns
    -------
    mat : (..., ndim+1, ndim+1) tensor
        Reconstructed affine matrix `mat = T @ Rx @ Ry @ Rz @ Z @ S`

    """
    def mat_last(mat):
        """Move matrix from first to last dimensions"""
        return mat.transpose([*range(2, mat.ndim), 0, 1])

    def vec_first(mat):
        """Move vector from last to first dimension"""
        return mat.transpose([-1, *range(mat.ndim-1)])

    # Prepare parameter vector
    if prm is not None:
        # A stacked tensor was provided: we must unstack it and extract
        # each component. We expect them to be ordered as [*T, *R, *Z, *S].
        prm = vec_first(np.asanyarray(prm))
        nb_prm, *batch_shape = prm.shape
        nb_t = ndim
        nb_r = ndim*(ndim-1) // 2
        nb_z = ndim
        nb_s = ndim*(ndim-1) // 2
        idx = 0
        prm_t = prm[idx:idx+nb_t] if nb_prm > idx else None
        idx = idx + nb_t
        prm_r = prm[idx:idx+nb_r] if nb_prm > idx else None
        idx = idx + nb_r
        prm_z = prm[idx:idx+nb_z] if nb_prm > idx else None
        idx = idx + nb_z
        prm_s = prm[idx:idx+nb_s] if nb_prm > idx else None
    else:
        # Individual components were provided, but some may be None, and
        # they might not have exactly the same batch shape (we only
        # require that their batch shapes can be broadcasted together).
        prm_t = np.asanyarray(translations if translations is not None else [0] * ndim)
        prm_r = np.asanyarray(rotations if rotations is not None else [0] * (ndim*(ndim-1)//2))
        prm_z = np.asanyarray(zooms if zooms is not None else [1] * ndim)
        prm_s = np.asanyarray(shears if shears is not None else [0] * (ndim*(ndim-1)//2))
        # Broadcast all batch shapes
        batch_shape = np.broadcast_shapes(prm_t.shape[:-1], prm_r.shape[:-1],
                                          prm_z.shape[:-1], prm_s.shape[:-1])
        prm_t = vec_first(np.broadcast_to(prm_t, batch_shape + prm_t.shape[-1:]))
        prm_r = vec_first(np.broadcast_to(prm_r, batch_shape + prm_r.shape[-1:]))
        prm_z = vec_first(np.broadcast_to(prm_z, batch_shape + prm_z.shape[-1:]))
        prm_s = vec_first(np.broadcast_to(prm_s, batch_shape + prm_s.shape[-1:]))

    backend = dict(dtype=prm_t.dtype)

    if ndim == 2:

        def make_affine(t, r, z, sh, o, i):
            if t is not None:
                T = [[i, o, t[0]],
                     [o, i, t[1]],
                     [o, o, i]]
                T = mat_last(np.asanyarray(T, **backend))
            else:
                T = np.eye(3, **backend)
                T = np.broadcast_to(T, [*batch_shape, 3, 3])
            if r is not None:
                c = np.cos(r)
                s = np.sin(r)
                R = [[c[0],  s[0], o],
                     [-s[0], c[0], o],
                     [o,     o,    i]]
                R = mat_last(np.asanyarray(R, **backend))
            else:
                R = np.eye(3, **backend)
                R = np.broadcast_to(R, [*batch_shape, 3, 3])
            if z is not None:
                Z = [[z[0], o,    o],
                     [o,    z[1], o],
                     [o,    o,    i]]
                Z = mat_last(np.asanyarray(Z, **backend))
            else:
                Z = np.eye(3, **backend)
                Z = np.broadcast_to(Z, [*batch_shape, 3, 3])
            if sh is not None:
                S = [[i, sh[0], o],
                     [o, i,     o],
                     [o, o,     i]]
                S = mat_last(np.asanyarray(S, **backend))
            else:
                S = np.eye(3, **backend)
                S = np.broadcast_to(S, [*batch_shape, 3, 3])

            return T.matmul(R.matmul(Z.matmul(S)))
    else:
        def make_affine(t, r, z, sh, o, i):
            if t is not None:
                T = [[i, o, o, t[0]],
                     [o, i, o, t[1]],
                     [o, o, i, t[2]],
                     [o, o, o, i]]
                T = mat_last(np.asanyarray(T, **backend))
            else:
                T = np.eye(4, **backend)
                T = np.broadcast_to(T, [*batch_shape, 4, 4])
            if r is not None:
                c = np.cos(r)
                s = np.sin(r)
                Rx = [[i, o,     o,    o],
                      [o, c[0],  s[0], o],
                      [o, -s[0], c[0], o],
                      [o, o,     o,    i]]
                Ry = [[c[1],  o, s[1], o],
                      [o,     i, o,    o],
                      [-s[1], o, c[1], o],
                      [o,     o,    o, i]]
                Rz = [[c[2],  s[2], o, o],
                      [-s[2], c[2], o, o],
                      [o,     o,    i, o],
                      [o,     o,    o, i]]
                Rx = mat_last(np.asanyarray(Rx, **backend))
                Ry = mat_last(np.asanyarray(Ry, **backend))
                Rz = mat_last(np.asanyarray(Rz, **backend))
                R = Rx.matmul(Ry.matmul(Rz))
            else:
                R = np.eye(4, **backend)
                R = np.broadcast_to(R, [*batch_shape, 4, 4])
            if z is not None:
                Z = [[z[0], o,    o,    o],
                     [o,    z[1], o,    o],
                     [o,    o,    z[2], o],
                     [o,    o,    o,    i]]
                Z = mat_last(np.asanyarray(Z, **backend))
            else:
                Z = np.eye(4, **backend)
                Z = np.broadcast_to(Z, [*batch_shape, 4, 4])
            if sh is not None:
                S = [[i, sh[0], sh[1], o],
                     [o, i,     sh[2], o],
                     [o, o,     i,     o],
                     [o, o,     o,     i]]
                S = mat_last(np.asanyarray(S, **backend))
            else:
                S = np.eye(4, **backend)
                S = np.broadcast_to(S, [*batch_shape, 4, 4])

            return T.matmul(R.matmul(Z.matmul(S)))

    zero = np.zeros([], **backend)
    zero = np.broadcast_to(zero, batch_shape)
    one = np.ones([], **backend)
    one = np.broadcast_to(one, batch_shape)

    # Build affine matrix
    mat = make_affine(prm_t, prm_r, prm_z, prm_s, zero, one)

    return mat


def matrix_to_lie(mat, *basis, max_iter=10000, tol=1e-9):
    """Compute the parameters of an affine matrix in a basis of the algebra.

    This function finds the matrix closest to ``mat`` (in the least squares
    sense) that can be encoded in the specified basis.

    Parameters
    ----------
    mat : (ndim+1, ndim+1) tensor_like
        Affine matrix

    basis : vector_like[basis_like]
        Basis of the Lie algebra(s).

    max_iter : int, default=10000
        Maximum number of Gauss-Newton iterations in the least-squares fit.

    tol : float, default=1e-9
        Tolerance criterion for convergence.

    Returns
    -------
    prm : tensor
        Parameters in the specified basis

    mat : (ndim+1, ndim+1) tensor
        Fitted matrix

    """

    # Authors
    # -------
    # .. John Ashburner <j.ashburner@ucl.ac.uk> : original GN fit in Matlab
    # .. Yael Balbastre <yael.balbastre@gmail.com> : Python code

    # Format mat
    mat = np.asanyarray(mat)
    backend = dict(dtype=mat.dtype)
    ndim = mat.shape[-1] - 1

    # Format basis
    basis = ensure_list(build_affine_basis(*basis, ndim, **backend))
    nb_basis = sum([len(b) for b in basis])

    def gauss_newton():
        # Predefine these values in case max_iter == 0
        n_iter = -1
        # Gauss-Newton optimisation
        prm = np.zeros(nb_basis, **backend)
        M = np.eye(ndim+1, **backend)
        norm = (mat ** 2).sum()
        crit = np.inf
        for n_iter in range(max_iter):

            # Compute derivative of each submatrix with respect to its basis
            # * Mi
            # * dMi/dBi
            Ms = []
            dMs = []
            hMs = []
            n_basis = 0
            for a_basis in basis:
                nb_a_basis = a_basis.shape[0]
                a_prm = prm[n_basis:(n_basis+nb_a_basis)]
                M, dM, hM = expm(a_prm, a_basis, grad_X=True, hess_X=True)
                # hM = hM.abs().sum(-3)  # diagonal majoriser
                Ms.append(M)
                dMs.append(dM)
                hMs.append(hM)
                n_basis += nb_a_basis
            M = np.stack(Ms)

            # Compute derivative of the full matrix with respect to each basis
            # * M = mprod(M[:, ...])
            # * dM/dBi = mprod(M[:i, ...]) @ dMi/dBi @ mprod(M[i+1:, ...])
            for n_mat, (dM, hM) in enumerate(zip(dMs, hMs)):
                if n_mat > 0:
                    pre = multi_dot(M[:n_mat])
                    dM = np.matmul(pre, dM)
                    hM = np.matmul(pre, hM)
                if n_mat < len(M)-1:
                    post = multi_dot(M[(n_mat+1):])
                    dM = np.matmul(dM, post)
                    hM = np.matmul(hM, post)
                dMs[n_mat] = dM
                hMs[n_mat] = hM
            dM = np.concatenate(dMs)
            hMs = [np.abs(hM).sum(-3) for hM in hMs]  # diagonal majoriser (Fessler)
            hM = np.concatenate(hMs)
            M = multi_dot(M)

            # Compute gradient/Hessian of the loss (squared residuals)
            diff = M - mat
            diff = diff.flatten()
            dM = dM.reshape((nb_basis, -1))
            hM = hM.reshape((nb_basis, -1))
            gradient = (dM*diff).sum(-1)
            hessian = np.matmul(dM, dM.T)
            hessian[np.arange(len(hessian)), np.arange(len(hessian))] += (hM * np.abs(diff)).sum(-1)
            hessian[np.arange(len(hessian)), np.arange(len(hessian))] *= 1.001
            delta_prm = np.matmul(np.linalg.inv(hessian), gradient)
            prm -= delta_prm

            # check convergence
            crit = np.dot(delta_prm.flatten(), delta_prm.flatten()) / norm
            if crit < tol:
                break

        if crit >= tol:
            warn('Gauss-Newton optimisation did not converge: '
                 'n_iter = {}, sos = {}.'.format(n_iter + 1, crit),
                 RuntimeWarning)

        return prm, M

    prm, M = gauss_newton()

    # TODO: should I stack parameters per basis?
    return prm, M


def matrix_to_classic(mat, return_stacked=True):
    """Compute the parameters of an affine matrix.

    This functions decomposes the input matrix into a product of
    simpler matrices (translation, rotation, ...) and extracts their
    parameters, so that the input matrix can be (approximately)
    reconstructed by `mat = T @ Rx @ Ry @ Rz @ Z @ S`

    This function only works in 2D and 3D.

    Parameters
    ----------
    mat : (..., ndim+1, ndim+1) tensor_like
        Affine matrix
    return_stacked : bool, default=True
        Return all parameters stacked in a vector

    Returns
    -------
    prm : (..., ndim*(ndim+1)) tensor, if return_stacked
        Individual parameters, ordered as
        [*translations, *rotations, *zooms, *shears].

    translations : (..., ndim) tensor, if not return_stacked
        Translation parameters.
    rotations : (..., ndim*(ndim-1)//2) tensor, if not return_stacked
        Rotation parameters, in radian.
    zooms : (..., ndim*) tensor, if not return_stacked
        Zoom parameters.
    shears : (..., ndim*(ndim-1)//2) tensor, if not return_stacked
        Shear parameters.

    """

    # Authors
    # -------
    # .. John Ashburner <j.ashburner@ucl.ac.uk> : original code (SPM12)
    # .. Stefan Kiebel <stefan.kiebel@tu-dresden.de> : original code (SPM12)
    # .. Mikael Brudfors <brudfors@gmail.com> : Python port
    # .. Yael Balbastre <yael.balbastre@gmail.com> : Batching + Autograd

    mat = np.asanyarray(mat)
    ndim = mat.shape[-1] - 1

    if ndim not in (2, 3):
        raise ValueError(f'Expected dimension 2 or 3, but got {ndim}.')

    # extract linear part + cholesky decomposition
    # (note that matlab's chol is upper-triangular by default while
    #  pytorch's is lower-triangular by default).
    #
    # > the idea is that M = R @ Z @ S and
    #   M.T @ M = (S.T @ Z.T @ R.T) @ (R @ Z @ S)
    #   M.T @ M = (S.T @ Z.T) @ (Z @ S)
    #           = U.T @ U
    #  where U is upper-triangular, such that the diagonal of U contains
    #  zooms and the off-diagonal elements of Z\U are the shears.
    lin = mat[..., :ndim, :ndim]
    chol = np.linalg.cholesky(lin.transpose(-1, -2).matmul(lin).H).H
    diag_chol = np.diagonal(chol, axis1=-1, axis2=-2)

    # Translations
    prm_t = mat[..., :ndim, -1]

    # Zooms (with fix for negative determinants)
    # > diagonal of the cholesky factor
    prm_z = diag_chol
    prm_z0 = np.where(lin.det() < 0, -prm_z[..., 0], prm_z[..., 0])
    prm_z0 = prm_z0[..., None]
    prm_z = np.concatenate((prm_z0, prm_z[..., 1:]), axis=-1)

    # Shears
    # > off-diagonal of the normalized cholesky factor
    chol = chol / diag_chol[..., None]
    upper_ind = np.triu_indices(chol.shape[-2], chol.shape[-1],  offset=1)
    prm_s = chol[..., upper_ind[0], upper_ind[1]]

    # Rotations
    # > we know the zooms and shears and therefore `Z @ S`.
    #   If the problem is well conditioned, we can recover the pure
    #   rotation (orthogonal) matrix as `R = M / (Z @ S)`.
    lin0 = classic_to_matrix(zooms=prm_z, shears=prm_s, ndim=ndim)
    lin0 = lin0[..., :ndim, :ndim]
    rot = lin.matmul(lin0.inverse())          # `R = M / (Z @ S)`
    clamp = lambda x: x.clamp(min=-1, max=1)  # correct rounding errors

    xz = rot[..., 0, -1]
    rot_y = np.asin(clamp(xz))
    if ndim == 2:
        prm_r = rot_y[..., None]
    else:
        xy = rot[..., 0, 1]
        yx = rot[..., 1, 0]
        yz = rot[..., 1, -1]
        xx = rot[..., 0, 0]
        zz = rot[..., -1, -1]
        zx = rot[..., -1, 0]
        cos_y = np.cos(rot_y)

        # find matrices for which the first rotation is 90 deg
        # (we cannot divide by its cos in that case)
        cos_zero = (np.abs(rot_y) - np.pi/2)**2 < 1e-9
        zero = xy.new_zeros([]).expand(cos_zero.shape)

        rot_x = np.where(cos_zero.bool(), zero,
                         np.atan2(clamp(yz/cos_y), clamp(zz/cos_y)))
        rot_z = np.where(cos_zero,
                         np.atan2(-clamp(yx), clamp(-zx/xz)),
                         np.atan2(clamp(xy/cos_y), clamp(xx/cos_y)))
        prm_r = np.stack((rot_x, rot_y, rot_z), axis=-1)

    if return_stacked:
        return np.concatenate((prm_t, prm_r, prm_z, prm_s), axis=-1)
    else:
        return prm_t, prm_r, prm_z, prm_s
