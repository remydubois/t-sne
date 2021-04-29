import math

import numpy as np
from numba import njit, prange
from tsne.neighbors.kdtree import KDTree
from tsne.neighbors.utils import rdist


@njit
def squaredist(x, y):
    return np.power(x - y, 2.0).sum(-1)


@njit(parallel=True)
def symetric_squaredist(x):
    """
    Computes mutual square distances computation.

    Parameters
    ----------
    x : np.Array
        Input points

    Returns
    -------
    np.Array
        Mutual square distances of shape (N, N)
    """
    P = np.empty((len(x), len(x)), dtype=np.float64)
    for i in prange(len(x)):
        for j in range(i, len(x)):
            if i == j:
                P[i, j] = 0.0
            else:
                P[i, j] = rdist(x[i], x[j])
            P[j, i] = P[i, j]
    return P


@njit
def p_squaredist(X):
    """
    Flattened mutual square distances computation in condensed format

    Parameters
    ----------
    X : np.array
        Input points

    Returns
    -------
    np.Array
        Square distances, of shape (N * (N - 1) // 2, )
    """
    N = len(X)
    n_pairs = N * (N - 1) // 2
    d = np.empty(n_pairs, dtype=np.float64)
    c = 0
    for i in range(N):
        for j in range(i + 1, N):
            d[c] = rdist(X[i], X[j])
            c += 1

    return d


@njit
def compute_entropy(p):
    """
    Naive Entropy computation:
    sum(p * log(p))

    Parameters
    ----------
    p : np.array
        One-dimensional distribution

    Returns
    -------
    Float
        Entropy
    """
    entropy = 0.0
    for i in range(p.shape[-1]):
        if p[i] > 0:
            entropy += p[i] * math.log(p[i])

    return -entropy


@njit
def bisect_decreasing_func(f, y0=0.0, x0=1.0, max_steps=100, tol=1.0e-2, verbose=0, *args):
    """
    Bisect a decreasing function.
    Finds the value of x0 so that f(x0) = y0.

    Parameters
    ----------
    f : Callable
        Function to bisect
    y0 : float, optional
        Desired value, by default 0.0
    x0 : float, optional
        Starting point, by default 1.0
    max_steps : int, optional
        Maximum steps authorized, by default 100
    tol : Float, optional
        Maximum allowed error, by default 1.0e-2
    verbose : int, optional
        Verbosity level, by default 0
    *args:
        Additional arguments passed to f during bisection

    Returns
    -------
    Float
        The final value found
    """
    binf = 0.0
    bsup = np.inf
    for step in range(max_steps):
        y = f(x0, *args)
        if np.isnan(y):
            break

        delta = y - y0
        error = abs(delta)
        if error < tol:
            return x0

        # If y is negative, then increase x
        if delta > 0:
            binf = x0
            if bsup == np.inf:
                x0 *= 2.0
            else:
                x0 += bsup
                x0 /= 2
        # Else decrease
        else:
            bsup = x0
            if bsup == -np.inf:
                x0 /= 2
            else:
                x0 += binf
                x0 /= 2
    # print(y - y0, error, tol)
    # raise ValueError('Reached max steps')
    if verbose > 0:
        print("Reached max steps")
    return x0


@njit
def compute_p_j_and_entropy(alpha, distances, i, P):
    """
    p_j computation.
    It modifies P in place so that the bisection process results in the desired P.

    Parameters
    ----------
    alpha : Float
        Scaling parameter. In practice, this should be 1 / (2 * sigma ** 2.)
    distances : np.array
        Mutual square distances
    neighbors : np.array
        Indices of closest neighbors
    i : Integer
        Index of the considered row
    P : np.Array
        Placeholder for the conditional probability matrix

    Returns
    -------
    Float
        Entropy
    """
    sigma_p_i = 0.0
    for j in range(len(distances)):
        if i == j:
            P[i, j] = 0.0
            continue

        num = distances[i, j]
        num *= alpha
        num = np.exp(-num)
        P[i, j] = num
        sigma_p_i += num

    P[i] /= sigma_p_i
    sigma_dist_p_i = np.sum(P[i] * distances[i])

    H = math.log(sigma_p_i) + sigma_dist_p_i * alpha

    return H


@njit
def compute_nn_p_j_and_entropy(alpha, distances, neighbors, i, P):
    """
    Approximates p_j computation by considering only nearest neighbbors.
    It modifies P in place so that the bisection process results in the desired P.

    Parameters
    ----------
    alpha : Float
        Scaling parameter. In practice, this should be 1 / (2 * sigma ** 2.)
    distances : np.array
        Mutual square distances
    neighbors : np.array
        Indices of closest neighbors
    i : Integer
        Index of the considered row
    P : np.Array
        Placeholder for the conditional probability matrix

    Returns
    -------
    Float
        Entropy
    """
    sigma_p_i = 0.0
    # distances, neighbors = tree.query_nearest(X[i], k)

    for d, j in zip(distances, neighbors):
        if j == i:
            P[i, j] = 0.0
            continue
        num = d * alpha
        num = np.exp(-num)
        P[i, j] = num
        sigma_p_i += num

    P[i] /= sigma_p_i
    sigma_dist_p_i = np.sum(P[i][neighbors] * distances)

    H = math.log(sigma_p_i) + sigma_dist_p_i * alpha

    return H


@njit(parallel=True, cache=False)
def compute_conditional_gaussian(
    X, desired_perplexity=10.0, tol=1.0e-2, max_steps=100, verbose=0, nn=-1
):
    """
    Conditional Gaussian probability matrix computation, using bisection search for perplexity setting.
    The bisection search process actually happens on the entropy, not perplexity.

    Parameters
    ----------
    X : np.Array
        Input points
    desired_perplexity : float, optional
        Desired perplexity (row-wise), by default 10.0
    tol : Float, optional
        Maximal tolerance error for the bisection search process, by default 1.0e-2
    max_steps : int, optional
        Maximal number of steps authorized for the bisection search process, by default 100
    verbose : int, optional
        Verbosity level, by default 0
    nn : int, optional
        Amount of nearest neighbors to approximat P on, by default -1
        If -1: all points are considered.

    Returns
    -------
    P
        The conditional gaussian probability matrix
    """

    H0 = np.log2(desired_perplexity)
    if nn > 0:
        tree = KDTree(X, 16)
        # distances, neighbors = tree.query_many_nearest(X)
    # To ensure that tol is right
    # tol = np.log2(tol)

    N = len(X)
    P = np.zeros((N, N), dtype=np.float64)
    if nn < 0:
        distances = symetric_squaredist(X)

    for i in prange(N):
        if 0 > nn:
            sigma_i = bisect_decreasing_func(
                compute_p_j_and_entropy, H0, 1.0, 100, tol, verbose, distances, i, P
            )
        else:
            neighbor_distances, neighbors = tree.query_nearest(X[i], nn)
            # neighbor_distances = symetric_squaredist(X)[i]
            # neighbors = np.arange(N)

            # import i<pdb; ipdb.set_trace()
            sigma_ = bisect_decreasing_func(
                compute_nn_p_j_and_entropy,
                H0,
                1.0,
                100,
                tol,
                verbose,
                neighbor_distances,
                neighbors,
                i,
                P,
            )

    return P


@njit
def compute_joint_gaussian(X, desired_perplexity=10.0, tol=1.0e-2, max_steps=100, verbose=0, nn=-1):
    """
    Joint Gaussian probability matrix computation using bisection search for perplexity setting.

    Parameters
    ----------
    X : np.Array
        Input points
    desired_perplexity : float, optional
        Desired perplexity for the modeled distribution, by default 10.0
    tol : Float, optional
        Perplexity tolerance, by default 1.0e-2
    max_steps : int, optional
        Maximum amount of bisection steps authorized, by default 100
    verbose : int, optional
        Verbostiy level, by default 0
    nn : int, optional
        Amount of nearest neighbors on which to compute P, by default -1
        If -1, all points are considered.

    Returns
    -------
    np.Array
        Joint probability matrix.
    """
    P = compute_conditional_gaussian(X, desired_perplexity, tol, max_steps, verbose, nn)
    P = P + P.T
    # in-place produces weird matrix
    # P += P.T
    P /= 2 * len(X)
    return P
