from tsne.conditional import compute_joint_gaussian
from tsne.metrics import trustworthiness
from tsne.gradients import exact_kl_gradient, exact_gradient_descent
from tsne.barnes_hut_gradients import barnes_hut_gradient_descent, compute_barnes_hut_gradients
from tsne.tsne import TSNE as NBTSNE
from tsne.utils import KL
from sklearn import datasets
from sklearn.manifold import TSNE as skTSNE, trustworthiness as sk_trustworthiness
import matplotlib.pyplot as plt
from numba import config
from timeit import Timer
import time
from numba import set_num_threads

set_num_threads(4)

import numpy as np

MACHINE_EPSILON = np.finfo(np.double).eps
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


def _kl_divergence(
    params, P, degrees_of_freedom, n_samples, n_components, skip_num_points=0, compute_error=True
):
    X_embedded = params.reshape(n_samples, n_components)

    # Q is a heavy-tailed distribution: Student's t-distribution
    dist = pdist(X_embedded, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.0
    dist **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)

    # Optimization trick below: np.dot(x, y) is faster than
    # np.sum(x * y) because it calls BLAS

    # Objective: C (Kullback-Leibler divergence of P and Q)
    if compute_error:
        kl_divergence = 2.0 * np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))
    else:
        kl_divergence = np.nan

    # Gradient: dC/dY
    # pdist always returns double precision distances. Thus we need to take
    grad = np.ndarray((n_samples, n_components), dtype=params.dtype)
    PQd = squareform((P - Q) * dist)

    for i in range(skip_num_points, n_samples):

        grad[i] = np.dot(np.ravel(PQd[i], order="K"), X_embedded[i] - X_embedded)

    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c
    # i = 0
    # P = squareform(P)
    # Q = squareform(Q)
    # dist = squareform(dist)
    # print('\n', grad[i], P[1, :3], Q[1, :3], dist[1, :3], PQd[1, :3], PQd.dtype)
    grad = grad.ravel()
    return kl_divergence, grad


def test_exact_kl_gradient():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1.0, (1_000, 3))
    y = rng.uniform(0, 1.0, (1_000, 2))

    P = compute_joint_gaussian(X)
    kl, grad = exact_kl_gradient(P, y)

    p = squareform(P)

    sk_kl, sk_grad = _kl_divergence(y.ravel(), p, 1, len(X), 2)
    np.testing.assert_allclose(sk_kl, kl, err_msg="KL differs")
    np.testing.assert_allclose(
        sk_grad.reshape(-1, 2), grad, err_msg="Gradients differ", atol=1.0e-3, rtol=0.0
    )
    print("Gradient value test passed")

    sk_time = Timer(lambda: _kl_divergence(y.ravel(), p, 1, len(X), 2)).timeit(100) / 100 * 1000
    nb_time = Timer(lambda: exact_kl_gradient(P, y)).timeit(100) / 100 * 1000
    assert nb_time <= sk_time * 1.1, "Numba significantly slower than sklearn"
    print(f"Gradient step speed test: {sk_time:.1f} (sk) versus {nb_time:.1f} (numba) ms")


def test_bh_gradient():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1.0, (50, 3))
    y = rng.uniform(0, 1.0, (50, 2))

    P = compute_joint_gaussian(X)
    bh_kl, bh_grad = compute_barnes_hut_gradients(P, y, theta=0.0)
    ex_kl, ex_grad = exact_kl_gradient(P, y)

    np.testing.assert_allclose(bh_kl, ex_kl, atol=1.0e-4, rtol=0.0, err_msg="KL differ")
    np.testing.assert_allclose(bh_grad, ex_grad, atol=1.0e-4, rtol=0.0, err_msg="grad differ")
    print("Gradient and KL value tests passed")

    # Mimic gradient descent to compare convergence
    lr = 10.0
    ttw_ex = []
    kl_ex = []
    y_ex = y.copy()
    for i in range(100):
        kl, ex_grad = exact_kl_gradient(P, y_ex)
        y_ex -= lr * ex_grad
        ttw_ex.append(trustworthiness(X, y_ex))
        kl_ex.append(kl)

    with plt.xkcd():
        f, ax = plt.subplots(figsize=(8, 6))
        ax.plot(ttw_ex, label="Exact", linestyle="-", linewidth=1)
        ax2 = ax.twinx()
        ax.set_ylabel("Trustworthiness")
        ax2.set_ylabel("KL Divergence")
        ax2.plot(kl_ex, linestyle="dotted")

        for theta in [0.5, 1.0, 2.0, 4.0]:
            ttw_bh = []
            kls = []
            y_bh = y.copy()
            for i in range(100):
                kl, bh_grad = compute_barnes_hut_gradients(P, y_bh, theta)
                y_bh -= lr * bh_grad
                ttw_bh.append(trustworthiness(X, y_bh))
                kls.append(kl)
            ax.plot(ttw_bh, label=f"BH, theta={theta}", linestyle="--", linewidth=1)
            ax2.plot(kls, linestyle="dotted")

        ax.legend()
        ax.set_title("Exact versus BH GD convergence")
        ax.set_xlabel("GD steps")
        plt.subplots_adjust(left=0.18, bottom=0.15, right=0.85, top=0.9)
        f.show()


def test_gradient_step_speed():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1.0, (5000, 3))
    y = rng.uniform(0, 1.0, (len(X), 2))

    P = compute_joint_gaussian(X)
    kl, grad = compute_barnes_hut_gradients(P, y, theta=0.0)
    _ = exact_kl_gradient(P, y)
    exact_time = Timer(lambda: exact_kl_gradient(P, y)).timeit(10) / 10 * 1000

    angles = np.arange(0.0, 4.0, 0.1)
    bh_times = []
    for theta in angles:
        bh_time = (
            Timer(lambda: compute_barnes_hut_gradients(P, y, theta=theta)).timeit(10) / 10 * 1000
        )
        if theta > 0.5:
            np.testing.assert_array_less(
                bh_time, exact_time, err_msg="Barnes-Hut approximation slower than exact"
            )
        bh_times.append(bh_time)
    # print(f"Gradient step speed test passed: {bh_time:.1f} ms (Barnes-Hut) versus {exact_time:.1f} ms (exact)")

    with plt.xkcd():
        f, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(f"BH versus exact GD step time ({len(X)} datapoints")
        ax.plot(angles, bh_times)
        ax.axhline(exact_time, linestyle="--")
        ax.set_xlabel("Angle")
        ax.set_ylabel("Gradient step time (ms)")
        plt.subplots_adjust(left=0.18, bottom=0.15, right=0.85, top=0.95)
        f.show()


def test_gradient_descent_trustworthiness():
    X = np.random.uniform(0, 1.0, (5_000, 3))
    y = np.random.uniform(0.0, 1.0, (len(X), 2))

    P = compute_joint_gaussian(X)

    y1 = y.copy()
    y2 = y.copy()
    barnes_hut_gradient_descent(
        P, y2, min_improvement=1.0e-5, theta=0.5, verbose=20, lr=50.0, n_steps=20
    )
    exact_gradient_descent(P, y1, min_improvement=1.0e-5, verbose=20, lr=50.0, n_steps=20)
    t1 = trustworthiness(X, y1)
    t2 = trustworthiness(X, y2)
    np.testing.assert_allclose(t1, t1, atol=3.0e-2, rtol=0.0)
    print(f"Trustworthiness test passed: {t1:.3f} (exact) versus {t2:.3f} (barnes-hut)")


def test_gradient_descent_speed():
    X = np.random.uniform(0, 1.0, (500, 3))
    y = np.random.uniform(0.0, 1.0, (500, 2))

    P = compute_joint_gaussian(X)

    y1 = y.copy()
    y2 = y.copy()
    exact_gradient_descent(P, y1, verbose=1)
    barnes_hut_gradient_descent(P, y2, theta=0.1, verbose=1)

    exact_time = Timer(lambda: exact_gradient_descent(P, y.copy(), verbose=0)).timeit(5) / 5
    bh_time = (
        Timer(lambda: barnes_hut_gradient_descent(P, y.copy(), verbose=0.0, theta=0.1)).timeit(5)
        / 5
    )

    np.testing.assert_array_less(bh_time, exact_time)
    print(f"Speed test passed: {exact_time:.3f} (exact) versus {bh_time:.3f} (Barnes-Hut)")


def test_trustworthiness():
    X = np.random.uniform(0, 1.0, (5_000, 3))
    y = np.random.uniform(0.0, 1.0, (5_000, 2))

    t = trustworthiness(X, y, k=4)
    t2 = sk_trustworthiness(X, y, n_neighbors=4)
    assert np.allclose(t, t2, atol=1.0e-3, rtol=0.0)
    print(f"Trustworthiness implementation test passed: {t:.3f} (own) versus {t2:.3f} (sklearn)")

    timer = Timer(lambda: trustworthiness(X, y, k=4))
    timer_sk = Timer(lambda: sk_trustworthiness(X, y, n_neighbors=4))

    repeats = 10
    time = timer.timeit(number=repeats) / repeats * 1000
    time_sk = timer_sk.timeit(number=repeats) / repeats * 1000
    print(f"Time (NB): {time:.3f} ms, (SK): {time_sk:.3f} ms")


def test_gradient_descent_sk_trustworthiness():
    X = np.random.uniform(0, 1.0, (500, 3))
    y = np.random.uniform(0.0, 1.0, (500, 2))

    y1 = (
        skTSNE(2, method="exact", learning_rate=10.0, init="random", verbose=10)
        .fit_transform(X)
        .astype(np.float64)
    )
    y2 = (
        skTSNE(2, method="barnes_hut", learning_rate=10.0, init="random", verbose=10)
        .fit_transform(X)
        .astype(np.float64)
    )
    t1 = trustworthiness(X, y1)
    t2 = trustworthiness(X, y2)
    np.testing.assert_allclose(t1, t1, atol=3.0e-2, rtol=0.0)
    print(f"Trustworthiness test passed: {t1:.3f} (exact) versus {t2:.3f} (barnes-hut)")
