from tsne.neighbors.kdtree import KDTree
from tsne.neighbors.quadtree import QuadTree
import numpy as np
from timeit import Timer
from numba import set_num_threads
import matplotlib.pyplot as plt
from sklearn.neighbors._quad_tree import _QuadTree
from threadpoolctl import threadpool_limits

# def test_neighbors():

#     X = np.random.default_rng(0).uniform(0., 1., (100, 2))
#     tree = KDTree(X)
#     _distances, indices = tree.query_nearest(X[0], 4)

#     distances = np.power(X[0] - X, 2.).sum(-1)
#     neighbors = distances.argsort()[:4]
#     assert np.allclose(indices, neighbors)


def test_many_neighbors_kdt():

    X = np.random.default_rng(0).uniform(0.0, 1.0, (100_000, 2))
    tree = KDTree(X)
    _distances, indices = tree.query_many_nearest(X[:], 4)
    repeats = 10

    for nt in [1, 2, 4, 8]:
        set_num_threads(nt)
        timer = Timer(lambda: tree.query_many_nearest(X[:], 4))
        delta = timer.timeit(number=repeats) / repeats
        print(nt, delta)


def test_many_resultants():

    X = np.random.default_rng(0).uniform(0.0, 1.0, (1_000, 2))
    tree = QuadTree(X)
    _distances, indices = tree.compute_many_resultants(0.1)
    repeats = 50
    print("")
    for nt in [1, 2, 4, 8]:
        print(f"---- {nt} threads ----")
        set_num_threads(nt)
        timer = Timer(lambda: tree.compute_many_resultants(0.1))
        delta = timer.timeit(number=repeats) / repeats
        print(nt, delta)


def test_quadtree_bulk():
    ns = np.arange(1_000, 1_000_000, 25_000)
    ts = []
    sk_ts = []
    repeats = 5

    for n in ns:
        X = np.random.default_rng(0).uniform(0.0, 1.0, (n, 2))
        _ = QuadTree(X)

        # Mine
        with threadpool_limits(1):
            timer = Timer(lambda: QuadTree(X, 16))
        delta = timer.timeit(number=repeats) / repeats * 1000
        ts.append(delta)

        # SK
        timer = Timer(lambda: _QuadTree(2, 0).build_tree(X))
        delta = timer.timeit(number=repeats) / repeats * 1000
        sk_ts.append(delta)

    with plt.xkcd():
        f, ax = plt.subplots()
        ax.plot(ns, ts, label="Mine", marker="o")
        ax.plot(ns, sk_ts, label="SK", marker="o")
        ax.set_ylabel("Elapsed time (ms)")
        ax.set_xlabel("n")
        ax.set_title(f"Quadtree bulkloading time (leaf_size=16)")
        ax.legend()
        f.show()


def test_quadtree_bh():
    ns = np.arange(1_000, 100_000, 5_000)
    ts = []
    sk_ts = []
    repeats = 5

    for n in ns:
        X = np.random.default_rng(0).uniform(0.0, 1.0, (n, 2))

        # Mine
        tree = QuadTree(X)
        tree.compute_resultant(X[0], 0.5)
        timer = Timer(lambda: tree.compute_resultant(X[0], 0.05))
        delta = timer.timeit(number=repeats) / repeats * 1000
        ts.append(delta)

        # SK
        # tree = _QuadTree(2, 0)
        # tree.build_tree(X)
        # import ipdb; ipdb.set_trace()
        # timer = Timer(lambda : tree.summarize(X[0]))
        timer = Timer(lambda: (1 / (1 + np.power(X[0] - X, 2.0).sum(-1))).sum())
        delta = timer.timeit(number=repeats) / repeats * 1000
        sk_ts.append(delta)

    with plt.xkcd():
        f, ax = plt.subplots()
        ax.plot(ns, ts, label="QuadTree", marker="o")
        ax.plot(ns, sk_ts, label="naive", marker="o")
        ax.set_ylabel("Elapsed time (ms)")
        ax.set_xlabel("n")
        ax.set_title(f"Quadtree resultant force computation time (leaf_size=16)")
        ax.legend()
        f.show()
