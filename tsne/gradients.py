import math
import time

import numpy as np
from numba import njit, objmode, prange
from tsne.conditional import p_squaredist, squaredist
from tsne.neighbors.utils import rdist
from tsne.utils import ravel_condensed_index

MACHINE_EPSILON = np.finfo(np.double).eps


@njit(parallel=False, cache=False)
def exact_kl_gradient(P, Y):
    """
    Compute gradients of the KL divergence.

    Parameters
    ----------
    P : np.Array
        Joint probability gaussian matrix
    Y : np.Array
        Embedded points

    Returns
    -------
    Tuple
        KL divergence value and its gradient
    """
    gradient = np.empty(Y.shape, dtype=Y.dtype)
    N = len(Y)

    # Q or d could be condensed
    d = np.empty((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i, N):
            if i == j:
                d[i, j] = 0.0
            else:
                d[i, j] = 1.0 / (1.0 + rdist(Y[i], Y[j]))
                # d[i, j] = rdist(Y[i], Y[j])
                d[j, i] = d[i, j]

    Z = 0.0
    for i in range(N - 1):
        for j in range(i + 1, N):
            Z += d[i, j]
    Z *= 2.0
    Q = d / Z

    kl = 0.0
    for i in range(N - 1):
        for j in range(i + 1, N):
            if P[i, j] == 0.0:
                continue
            kl += P[i, j] * math.log(P[i, j] / Q[i, j])
    kl *= 2.0

    grad = np.empty(Y.shape, dtype=Y.dtype)
    for i in range(N):
        p_qd = (P[i] - Q[i]) * d[i]
        g = np.dot(p_qd, Y[i] - Y) * 4.0

        for k in range(Y.shape[-1]):
            grad[i, k] = g[k]

    return kl, grad


@njit(cache=True)
def exact_gradient_descent(
    P, Y, lr=10.0, momentum=0.8, n_steps=1000, patience=5, min_improvement=1.0e-4, verbose=0
):

    """
    Gradient descent for the KL divergence with momentum.

    Parameters
    ----------
    P : np.array
        Gaussian joint probability matrix
    Y : np.array
        Embedded points to optimize
    lr : float, optional
        Learning rate, by default 1.0
    momentum : float, optional
        Momentum, by default 0.8
    n_steps : int, optional
        Max amount of steps, by default 10000
    patience : int, optional
        Patience, by default 5
    min_improvement : [type], optional
        Minimal admissible improvement, by default 1.0e-4
    verbose : int, optional
        Verbosity level, by default 0

    Returns
    -------
    np.Array
        The final, optimized parameters Y
    """
    upd = np.zeros_like(Y)
    best = 1.0e9
    buffer = 0

    for i in range(n_steps):
        if verbose >= 2:
            with objmode(tic="f8"):
                tic = time.time()
        kl, grad = exact_kl_gradient(P, Y)

        if verbose >= 2:
            with objmode(toc="f8"):
                toc = time.time()

        if verbose > 1:
            if i == 0:
                print("[TSNE] Gradient Descent Initial KL:", kl)

        if kl < best - min_improvement:
            best = kl
            buffer = 0
        else:
            buffer += 1
            if buffer == patience:
                if verbose >= 2:
                    print(
                        "[TSNE] Gradient Descent Improvement smaller than",
                        min_improvement,
                        ", interrupting gradient descent. Final KL",
                        kl,
                    )
                break

        if i == 0.0:
            upd = -lr * grad
        else:
            upd = 0.8 * upd - lr * grad

        Y += upd

        if verbose >= 2:
            if i % verbose == 0:
                with objmode():
                    print(
                        "[TSNE] Gradient Descent Step {0}, KL: {1:.4f}, step cpu time: {2:.1f} ms".format(
                            i, kl, (toc - tic) * 1000
                        )
                    )
    if verbose >= 1:
        print("[TSNE] Gradient Descent Final KL:", kl)
    return Y
