from tsne.neighbors.kdtree import KDTree
from tsne.neighbors.quadtree import Node, QuadTree
from tsne.neighbors.utils import quadsplit
import numpy as np
from timeit import Timer

from numba import set_num_threads

set_num_threads(1)


def test_quad_creation():
    X = np.random.default_rng(0).uniform(0.0, 1.0, (100, 2))

    tree = QuadTree(X)


def test_quad_neighbors():

    X = np.random.default_rng(0).uniform(0.0, 1.0, (100, 3))

    tree = QuadTree(X)
    _distances, indices = tree.query_nearest(X[0], 4)
    distances = np.power(X[0] - X, 2.0).sum(-1)
    neighbors = distances.argsort()[:4]
    assert np.allclose(indices, neighbors)


def test_quadsplit():
    X = np.random.default_rng(0).uniform(0.0, 1.0, (100_000, 12))
    quadsplit(X)
    repeats = 10
    timer = Timer(lambda: quadsplit(X))
    print("\n", timer.timeit(number=repeats) / repeats * 1000, "ms for", len(X), "points")


def test_resultant():
    X = np.random.default_rng(0).uniform(0.0, 1.0, (10_000, 2))

    # Compute exact
    exact = np.power(X[0] - X, 2.0).sum(-1)
    # exact = (1 / (1 + distances)).sum()

    # Approx
    tree = QuadTree(X, leaf_size=16)
    distances, indices = tree.compute_resultant(X[0], 0.0)
    import ipdb

    ipdb.set_trace()

    assert np.allclose(exact.sum(), distances.sum(), atol=1.0e-3, rtol=0.0)


def test_kd_neighbors():

    X = np.random.default_rng(0).uniform(0.0, 1.0, (100, 2))
    tree = KDTree(X)
    _distances, indices = tree.query_nearest(X[0], 4)

    distances = np.power(X[0] - X, 2.0).sum(-1)
    neighbors = distances.argsort()[:4]
    assert np.allclose(indices, neighbors)


def test_kd_many_neighbors():

    X = np.random.default_rng(0).uniform(0.0, 1.0, (1_000, 4))
    tree = KDTree(X)
    _distances, indices = tree.query_many_nearest(X[0:5], 4)

    for i in range(5):
        distances = np.power(X[i] - X, 2.0).sum(-1)
        neighbors = distances.argsort()[:4]
        assert np.allclose(indices[i], neighbors)
    print("KD Tree NN test passed")

    # Speed
    for nt in [1, 2, 4, 8]:
        set_num_threads(nt)
        delta = Timer(lambda: tree.query_many_nearest(X, 4)).timeit(10) / 10 * 1000
        print(f"Num threads: {nt}, query time on {len(X)} {X.shape[-1]}-d points: {delta:.2f} ms ")
