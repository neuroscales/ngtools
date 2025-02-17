"""Implementation of the matrix exponential and its derivative.

This implementation is based on John Ashburner's in SPM, which relies
on a Taylor approximation for both the exponential and its derivative.
Faster implementations that rely on scaling and squaring or Pade
approximations (as in scipy) could be used instead. This may be the
object of future work.

"""
# externals
import numpy as np


def expm(X, basis=None, grad_X=False, grad_basis=False, hess_X=False,
         max_order=10000, tol=1e-32):
    """Matrix exponential (and its derivatives).

    Parameters
    ----------
    X : {(..., F), (..., D, D)} array
        If `basis` is None: log-matrix.
        Else:               parameters of the log-matrix in the basis set.
    basis : (..., F, D, D) array, default=None
        Basis set. If None, basis of all DxD matrices and F = D**2.
    grad_X : bool, default=False
        Compute derivatives with respect to `X`.
    grad_basis : bool, default=False
        Compute derivatives with respect to `basis`.
    max_order : int, default=10000
        Order of the Taylor expansion
    tol : float, default=1e-32
        Tolerance for early stopping
        The criterion is based on the Frobenius norm of the last term of
        the Taylor series.

    Returns
    -------
    eX : (..., D, D) tensor
        Matrix exponential
    dX : (..., F, D, D) tensor, if `grad_X is True`
        Derivative of the matrix exponential with respect to the
        parameters in the basis set
    dB : (..., F, D, D, D, D) tensor, if `grad_basis is True`
        Derivative of the matrix exponential with respect to the basis.
    hX : (..., F, F, D, D) tensor, if `hess_X is True`
        Second derivative of the matrix exponential with respect to the
        parameters in the basis set

    """

    # Authors
    # -------
    # .. John Ashburner <j.ashburner@ucl.ac.uk> : original Matlab code
    # .. Mikael Brudfors <brudfors@gmail.com> : Python port
    # .. Yael Balbastre <yael.balbastre@gmail.com> : Python port
    #
    # License
    # -------
    # The original Matlab code is (C) 2012-2019 WCHN / John Ashburner
    # and was distributed as part of [SPM](https://www.fil.ion.ucl.ac.uk/spm)
    # under the GNU General Public Licence (version >= 2).

    def smart_incr(a, b):
        """Use inplace or outplace increment based on `a` and `b`s shapes."""
        if a.shape == b.shape:
            a += b
        else:
            a = a + b
        return a

    X = np.asanyarray(X)

    if basis is not None:
        # X contains parameters in the Lie algebra -> reconstruct the matrix
        # X.shape = [.., F], basis.shape = [..., F, D, D]
        basis = np.asanyarray(basis, dtype=X.dtype)
        param = X
        X = np.sum(basis * X[..., None, None], axis=-3, keepdims=True)
        dim = basis.shape[-1]
    else:
        # X contains matrices in log-space -> build basis set
        # X.shape = [..., D, D]
        dim = X.shape[-1]
        param = X.reshape(X.shape[:-2] + (-1,))
        basis = np.eye(dim * dim, dtype=X.dtype)
        basis = basis.reshape([dim**2, dim, dim])
        X = X[..., None, :, :]

    XB_batch_shape = X.shape[:-3]
    nb_feat = param.shape[-1]

    if grad_basis:
        # Build a basis for the basis
        basis_basis = np.eye(dim * dim, dtype=X.dtype)
        basis_basis = basis_basis.reshape([1, dim, dim, dim, dim])
        basis_basis = basis_basis * param[..., None, None, None, None]
        basis_basis = basis_basis.reshape(XB_batch_shape + (-1, dim, dim))

    # At this point:
    #   X.shape           = [*XB_batch_shape, 1, D, D]
    #   basis.shape       = [*B_batch_shape,  F, D, D]
    #   param.shape       = [*X_batch_shape,  F]
    #   basis_basis.shape = [*XB_batch_shape, F*D*D, D, D]

    # Aliases
    I = np.eye(dim, dtype=X.dtype)
    E = I + X                            # expm(X)
    En = np.copy(X)                      # n-th Taylor coefficient of expm
    if grad_X or hess_X:
        dE = np.copy(basis)              # dexpm(X)/dx
        dEn = np.copy(basis)             # n-th Taylor coefficient of dexpm/dx
    if grad_basis:
        dB = np.copy(basis_basis)        # dexpm(X)/dB
        dBn = np.copy(basis_basis)       # n-th Taylor coefficient of dexpm/dB
    if hess_X:
        hE = np.zeros_like(E, shape=[*XB_batch_shape, nb_feat, nb_feat, dim, dim])
        hEn = np.copy(hE)

    for n_order in range(2, max_order+1):
        # Compute coefficients at order `n_order`, and accumulate
        if hess_X:
            dEB = np.matmul(dEn[..., None, :, :, :], basis[..., None, :, :])
            hEn = np.matmul(hEn, X[..., None, :, :]) + dEB + dEB.swapaxes(-3, -4)
            hEn /= n_order
            hE = smart_incr(hE, hEn)
            del dEB
        if grad_X:
            dEn = np.matmul(dEn, X) + np.matmul(En, basis)
            dEn /= n_order
            dE = smart_incr(dE, dEn)
        if grad_basis:
            dBn = np.matmul(dBn, X) + np.matmul(En, basis_basis)
            dBn /= n_order
            dB = smart_incr(dB, dBn)
        En = np.matmul(En, X)
        En /= n_order
        E = smart_incr(E, En)
        # Compute sum-of-squares
        sos = np.sum(En * En)
        if sos <= En.size * tol:
            break

    E = E[..., 0, :, :]
    if grad_basis:
        dB = dB.reshape(XB_batch_shape + (nb_feat, dim, dim, dim, dim))

    out = [E]
    if grad_X:
        out.append(dE)
    if grad_basis:
        out.append(dB)
    if hess_X:
        out.append(hE)
    return out[0] if len(out) == 1 else tuple(out)
