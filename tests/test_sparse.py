from tsne.sparse import SparseArray
from scipy.sparse import coo_matrix
import numpy as np


def test_init():
    rng = np.random.default_rng(0)
    indices = rng.integers(0, 100, size=(10_000, 2))
    values = rng.uniform(size=(len(indices),))

    sp = SparseArray(indices, values)


def test_dense():

    rng = np.random.default_rng(0)
    indices = rng.integers(0, 100, size=(100, 2))
    # Ensure it covers [0; 100]
    indices[np.array([0, 99])] = np.array([[0, 0], [99, 99]], dtype=indices.dtype)
    values = rng.uniform(size=(len(indices),))

    # sp = SparseArray(indices, values).dense((100, 100))
    sp = SparseArray(indices, values).dense()

    t = np.zeros((100, 100))
    t[tuple(indices.T)] = values
    np.testing.assert_allclose(t, sp)


def test_transpose():
    rng = np.random.default_rng(0)
    indices = rng.integers(0, 100, size=(100, 2))
    values = rng.uniform(size=(len(indices),))

    sp = SparseArray(indices, values).T.dense((100, 100))

    t = np.zeros((100, 100))
    t[tuple(indices.T)] = values
    t = t.T

    np.testing.assert_allclose(t, sp)


def test_add():
    rng = np.random.default_rng(0)
    t = np.zeros((100, 100))

    indices = rng.integers(0, 100, size=(100, 2))
    values = rng.uniform(size=(len(indices),))
    # indices = np.array([[0, 1]])
    # values = np.array([1.])
    sp1 = SparseArray(indices, values)
    t[tuple(indices.T)] += values

    indices = rng.integers(0, 100, size=(100, 2))
    values = rng.uniform(size=(len(indices),))
    sp2 = SparseArray(indices, values)
    t[tuple(indices.T)] += values

    # Check naive add
    d = sp2.add(sp1).dense((100, 100))
    np.testing.assert_allclose(t, d)

    # Check symetry
    d2 = sp1.add(sp2).dense((100, 100))
    np.testing.assert_allclose(d2, d)


def test_indexing():
    rng = np.random.default_rng(0)
    t = np.zeros((100, 100))

    indices = rng.integers(0, 100, size=(100, 2))
    values = rng.uniform(size=(len(indices),))
    # indices = np.array([[0, 1]])
    # values = np.array([1.])
    sp1 = SparseArray(indices, values)
    t[tuple(indices.T)] += values

    for i in range(len(t)):
        # get non null values
        target = t[i][np.where(t[i])]
        v = sp1[i]
        # compare
        np.testing.assert_allclose(target, v)
