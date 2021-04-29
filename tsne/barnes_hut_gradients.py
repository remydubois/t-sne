import math
import time

import numpy as np
from numba import njit, objmode, prange
from tsne.neighbors.quadtree import QuadTree
from tsne.neighbors.utils import rdist


@njit
def add_F_attr(i, t_distances, P, Y, grad, Z):
    """
    Account for the positive forces of the KL divergence gradient approximation.

    Parameters
    ----------
    i : Integer
        Index of the point to compute gradient for
    t_distances : np.array
        Student-kernelized distances.
    P : np.array
        Joint probabilities matrix
    Y : np.array
        Embedded points
    grad : np.array
        Placeholder for gradient
    Z : np.array
        Placeholder for Z

    Returns
    -------
    Float
        The kl divergence
    """
    kl = 0.0

    grad[:] = 0.0
    for j in range(len(Y)):
        if i == j or P[i, j] == 0:
            # for k in range(Y.shape[-1]):
            # grad[:] = 0.
            continue
        wij = P[i, j] * t_distances[i, j]
        for k in range(Y.shape[-1]):
            grad[k] += (Y[i, k] - Y[j, k]) * wij

        kl += P[i, j] * np.log(P[i, j] / t_distances[i, j] * Z)
        # if not np.isfinite(kl) or not np.isfinite(grad).all():
        #     print('Overflow =>', P[i, j], t_distances[i, j], Z, P[i, j] / t_distances[i, j] * Z, np.log(P[i, j] / t_distances[i, j] * Z))
        #     raise ValueError('Overflow')
    return kl


@njit(parallel=True)
def compute_barnes_hut_gradients(P, Y, theta=0.01):
    """
    Compute gradients under the barnes-hut approximation.

    Parameters
    ----------
    P : np.Array
        Joint probability gaussian matrix
    Y : np.Array
        Embedded points
    theta : float, optional
        Limit angle, by default 0.01

    Returns
    -------
    Tuple
        KL divergence value and its gradient
    """
    N = len(Y)

    tree = QuadTree(Y, 16)

    # In theory this is O(u*n*log(n)) since P is sparse
    # It should be faster to use explicitely sparse array
    t_distances = np.empty((N, N), dtype=np.float64)
    for i in prange(N):
        for j in range(i + 1, N):
            if P[i, j] == 0:
                t_distances[i, j] = 0.0
                t_distances[j, i] = 0.0
            else:
                t_distances[i, j] = 1 / (1.0 + rdist(Y[i], Y[j]))
                t_distances[j, i] = t_distances[i, j]

    # My understanding is that distances of embedded points is computed in t_distances
    # and barnes hut but not on the same points: t_dist is only knn in the input space
    # Barnes Hut to approximate Z and resultant forces
    zF_rep, Z = tree.compute_many_resultants(theta)

    grad = np.empty_like(Y)
    kl = 0.0
    for i in prange(N):
        kl += add_F_attr(i, t_distances, P, Y, grad[i], Z)
        for k in range(Y.shape[-1]):
            grad[i, k] -= zF_rep[i, k] / Z

    grad *= 4.0

    return kl, grad


@njit
def barnes_hut_gradient_descent(
    P,
    Y,
    theta=0.1,
    lr=10.0,
    momentum=0.8,
    n_steps=1000,
    patience=5,
    min_improvement=1.0e-4,
    verbose=0,
):
    """
    Gradient descent for the KL divergence using the Barnes Hut approximation with momentum.

    Parameters
    ----------
    P : np.array
        Gaussian joint probability matrix
    Y : np.array
        Embedded points to optimize
    theta : float, optional
        Limit angle, by default 0.1
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
        kl, grad = compute_barnes_hut_gradients(P, Y, theta)
        if verbose >= 2:
            with objmode(toc="f8"):
                toc = time.time()

        if verbose > 1:
            if i == 0:
                print("[TSNE] BH Gradient Descent Initial KL:", kl)

        if kl < best - min_improvement:
            best = kl
            buffer = 0
        else:
            buffer += 1

            if buffer == patience:
                if verbose >= 2:
                    print(
                        "[TSNE] BH Gradient Descent Improvement smaller than",
                        min_improvement,
                        ", interrupting gradient descent.",
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
                        "[TSNE] BH Gradient Descent Step {0}, KL: {1:.4f}, step cpu time: {2:.1f} ms".format(
                            i, kl, (toc - tic) * 1000
                        )
                    )  # , end='\r')
    if verbose >= 1:
        print("[TSNE] BH Gradient Descent Final KL:", kl)
    return Y
