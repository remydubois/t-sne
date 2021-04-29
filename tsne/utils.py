from numba import int64, njit, vectorize


# @vectorize([int64(int64, int64, int64)], target='parallel')
@njit
def ravel_condensed_index(i, j, N):
    if i == j:
        raise ValueError("Diagonal out of bounds")
    if i > N - 1 or j > N - 1:
        raise ValueError("Out of bounds")

    i, j = min(i, j), max(i, j)

    L = N - 1
    k = 0
    for r in range(i):
        k += L - r
    k += j - (i + 1)

    # k = 0
    # for r in range(i + 1):
    #     for l in range(r + 1, j + 1):
    #         k += 1
    # # Note that we counted twice diagonal element
    # k -= (i + 1)

    return int(k)


def KL(P, Q):
    kl = np.where(P != 0.0 & Q != 0.0, P * np.log(P / Q), 0.0)
    return kl.sum(-1, -2)
