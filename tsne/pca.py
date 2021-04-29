import numpy as np
from numba import njit, prange


@njit(cache=True)
def project_pca(x, n_components=2):
    """
    Minimalistic PCA.
    Note that it does not guarantee a deterministic output because of SGD decomposition,
    which is sign-unaware.

    Parameters
    ----------
    x : np.array
        Input data
    n_components : int, optional
        Number of components to keep, by default 2

    Returns
    -------
    np.array
        Projected array
    """
    # Normalize
    means = np.empty(x.shape[-1], dtype=x.dtype)
    for i in range(x.shape[-1]):
        means[i] = x[:, i].mean()
    x_norm = x - means

    # Compute cov
    cov = np.cov(x_norm, rowvar=False)
    # Get eig
    eigval, eigvec = np.linalg.eig(cov)
    # Sort eigvals
    idxs = np.argsort(eigval)[::-1][:n_components]
    eigval = eigval[idxs]
    eigvec = eigvec[:, idxs]

    # Project
    y = np.dot(eigvec.T, x_norm.T).T

    return y
