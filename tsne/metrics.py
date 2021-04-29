import numpy as np
from numba import njit, prange
from tsne.conditional import symetric_squaredist
from tsne.neighbors.kdtree import KDTree


@njit(parallel=True)
def trustworthiness(X, Y, k=16):
    """
    Trustworthiness metric. It penalizes when points far away in the
    original space finds themselves close in the embedded space.
    1. is perfect, 0. is the worst.

    Parameters
    ----------
    X : np.Array
        Input points
    Y : np.Array
        Embedded points
    k : int, optional
        Amount of neighbors to limit computations to, by default 16

    Returns
    -------
    Float
        The trustworthiness score
    """
    output_space_tree = KDTree(Y, 16)

    distances = symetric_squaredist(X)

    N = len(X)
    ranks = 0
    for i in prange(N):
        natural_neighbors = np.argsort(distances[i])
        _, N_k_i = output_space_tree.query_nearest(Y[i], k + 1)
        # Skip self as a neighbor
        for j in N_k_i[1:]:
            # Find the preferential position of the embedded neighbor in the natural space
            # I assumed that in the formula, position starts at 1, human way to read positions.
            r_i_j = 1
            # Skip self as a neighbor
            for l in natural_neighbors[1:]:
                # If we found the position, stop counting
                if l == j:
                    break
                # Else, increase position counter
                r_i_j += 1

            ranks += max(0, r_i_j - k)

    ranks *= 2
    ranks /= N * k * (2 * N - 3 * k - 1)
    ranks *= -1.0
    ranks += 1
    return ranks
