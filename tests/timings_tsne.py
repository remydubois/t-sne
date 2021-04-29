from tsne.conditional import compute_joint_gaussian
from tsne.metrics import trustworthiness
from tsne.gradients import exact_kl_gradient, exact_gradient_descent
import numpy as np
from timeit import Timer
from tsne.tsne import TSNE as NBTSNE
from sklearn import datasets
from sklearn.manifold import TSNE as skTSNE, trustworthiness as sk_trustworthiness
import matplotlib.pyplot as plt
import tqdm
import time


def test_tsne():
    X = np.random.uniform(0, 1.0, (1000, 3))
    y = np.random.uniform(0.0, 1.0, (len(X), 2))

    P = compute_joint_gaussian(X)
    kl, grad = exact_kl_gradient(P, y)

    timer = Timer(lambda: exact_kl_gradient(P, y))
    delta = timer.timeit(number=100) / 100 * 1000
    print(f"Timing grad step f{delta:.2f} ms")


def test_iris():
    iris = datasets.load_iris()
    tiling = 1
    X = iris.data.repeat(tiling, axis=0)
    t = iris.target.repeat(tiling, axis=0)
    verbose = 0

    _ = NBTSNE(2.0, perplexity=10, lr=10.0, verbose=verbose, nn=-1, method="exact").fit_transform(X)
    tic = time.time()
    y_nb = NBTSNE(2.0, perplexity=10, lr=10.0, nn=64, verbose=verbose).fit_transform(X)
    print("\nNB timing", time.time() - tic)
    tic = time.time()
    y_sk = skTSNE(
        2, perplexity=10.0, init="random", verbose=verbose, method="exact", early_exaggeration=12.0
    ).fit_transform(X)
    print("SK timing", time.time() - tic)
    y_sk = y_sk.astype(np.float64)

    T_nb = trustworthiness(X, y_nb, k=16)
    T_sk = trustworthiness(X, y_sk, k=16)

    with plt.xkcd():
        f, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 4))
        ax0.scatter(*y_nb.T, c=t, cmap=plt.cm.Set1, edgecolor="k", s=40)
        ax0.set_title(f"Trustworthiness: {T_nb:.3f}", c="k")
        ax1.scatter(*y_sk.T, c=t, cmap=plt.cm.Set1, edgecolor="k", s=40)
        ax1.set_title(f"Trustworthiness: {T_sk:.3f}", c="k")
        ax0.set_xticks([])
        ax0.set_xlabel("Numba")
        ax1.set_xlabel("sklearn")
        ax0.set_yticks([])
        ax1.set_xticks([])
        ax1.set_yticks([])
        f.show()


def test_bh_tsne():
    rng = np.random.default_rng(0)

    ns = np.arange(500, 1_000, 500)
    nb_exact_ts = []
    nb_bh_ts = []
    sk_exact_ts = []
    sk_bh_ts = []
    repeats = 5
    verbose = 1

    for N in tqdm.tqdm(ns):
        x = rng.uniform(0.0, 1.0, size=(N, 4))

        exact_nbtsne = NBTSNE(2.0, perplexity=10, lr=10.0, verbose=verbose, nn=-1, method="exact")
        bh_nbtsne = NBTSNE(2.0, perplexity=10, lr=10.0, verbose=verbose, nn=-1, method="bh")
        exact_sk_tsne = skTSNE(
            2,
            perplexity=10.0,
            init="random",
            verbose=verbose,
            method="exact",
            early_exaggeration=1.0,
        )
        bh_sk_tsne = skTSNE(
            2,
            perplexity=10.0,
            init="random",
            verbose=verbose,
            method="barnes_hut",
            early_exaggeration=1.0,
        )
        _ = bh_nbtsne.fit_transform(x)
        _ = exact_nbtsne.fit_transform(x)

        timer_exact_nb = Timer(lambda: exact_nbtsne.fit_transform(x))
        timer_bh_nb = Timer(lambda: bh_nbtsne.fit_transform(x))
        timer_exact_sk = Timer(lambda: exact_sk_tsne.fit_transform(x))
        timer_bh_sk = Timer(lambda: bh_sk_tsne.fit_transform(x))

        nb_exact_ts.append(timer_exact_nb.timeit(repeats) / repeats * 1000)
        nb_bh_ts.append(timer_bh_nb.timeit(repeats) / repeats * 1000)
        sk_exact_ts.append(timer_exact_sk.timeit(repeats) / repeats * 1000)
        sk_bh_ts.append(timer_bh_sk.timeit(repeats) / repeats * 1000)

    with plt.xkcd():
        f, ax = plt.subplots()
        ax.plot(ns, nb_exact_ts, marker="o")
        ax.plot(ns, nb_bh_ts, marker="o")
        ax.plot(ns, sk_exact_ts, marker="o")
        ax.plot(ns, sk_exact_ts, marker="o")
        f.show()
