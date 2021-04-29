from tsne.conditional import (
    compute_joint_gaussian,
    compute_entropy,
    p_squaredist,
    compute_conditional_gaussian,
)
from scipy.spatial.distance import pdist
from tsne.utils import ravel_condensed_index
import numpy as np

from timeit import Timer


def test_complete_conditional_gaussian():
    X = np.random.uniform(0, 1.0, (5_000, 3))
    y = np.random.uniform(0.0, 1.0, (5_000, 2))

    desired_pp = 30.0
    tol = 1.0e-2
    P = compute_conditional_gaussian(X, desired_perplexity=desired_pp, tol=tol)

    H = np.array([compute_entropy(p_i) for p_i in P])
    np.testing.assert_allclose(2 ** H, desired_pp, rtol=0.0, atol=2 ** tol)


def test_nn_conditional_gaussian():
    X = np.random.uniform(0, 10.0, (5_000, 32))
    y = np.random.uniform(0.0, 10.0, (5_000, 2))

    desired_pp = 30.0
    tol = 1.0e-2
    P = compute_conditional_gaussian(
        X, desired_perplexity=desired_pp, tol=tol, nn=int(5 * desired_pp)
    )
    _ = compute_conditional_gaussian(X, desired_perplexity=desired_pp, tol=tol, nn=-1)

    H = np.array([compute_entropy(p_i) for p_i in P])
    np.testing.assert_allclose(2 ** H, desired_pp, rtol=0.0, atol=2 ** tol)
    print(
        f"KNN Perplexity test passed: [{2 ** H.min()} - {2 ** H.max()}](nn approx) versus {desired_pp}"
    )

    ex_timer = (
        Timer(
            lambda: compute_conditional_gaussian(X, desired_perplexity=desired_pp, tol=tol, nn=-1)
        ).timeit(50)
        / 50
        * 1000
    )
    nn_timer = (
        Timer(
            lambda: compute_conditional_gaussian(
                X, desired_perplexity=desired_pp, tol=tol, nn=int(5 * desired_pp)
            )
        ).timeit(50)
        / 50
        * 1000
    )

    np.testing.assert_array_less(nn_timer, ex_timer, err_msg="NN appprox slower than exact")
    print(
        f"NN-Approx conditional Gaussian speed test passed: {nn_timer:.2f} (nn) versus {ex_timer:.2f} (exact) ms"
    )


def test_p_dist():
    X = np.random.uniform(0, 1.0, (5_000, 3))
    a = p_squaredist(X)
    b = pdist(X, "sqeuclidean")
    np.testing.assert_allclose(a, b)


def test_raveled_pdist():
    N = 5_000
    X = np.random.uniform(0, 1.0, (N, 3))
    a = p_squaredist(X)
    full_dist = np.power(X[None] - X[:, None], 2.0).sum(-1)

    i = np.random.randint(0, N - 1)
    j = np.random.randint(0, N - 1)
    # Ensure that no diagonal is hit
    j = np.where(i == j, i + 1, j)
    k = ravel_condensed_index(i, j, N)
    np.testing.assert_allclose(a[k], full_dist[i, j], atol=1.0e-4, rtol=0.0)
