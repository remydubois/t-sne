from collections import OrderedDict
from typing import Tuple

import numpy as np
from numba import float32, float64, int64, njit, optional, prange
from numba.experimental import jitclass

specs = OrderedDict()
specs["indices"] = int64[:, :]
specs["N"] = int64
specs["dimensionality"] = int64
specs["values"] = float64[:]
specs["default_value"] = float64
specs["_row_indexer"] = int64[:]


@jitclass(specs)
class SparseArray:
    """

    A class for 2d sparse arrays. It support only what I need: transposition, addition.

    Disclaimer: basic features such as indices unicity tests are missing.
    If indices are non unique, addition will probably fail.
    """

    def __init__(self, indices, values, default_value=0.0, already_sorted=False):
        self.default_value = default_value

        if not len(indices) == len(values):
            raise ValueError("Uneven amount of coordinates and values")

        self.indices = indices
        self.values = values

        if not already_sorted:
            self.indices, self.values = sort_indices(self.indices, self.values)

        self.N = len(indices)
        self.dimensionality = indices.shape[-1]
        self._row_indexer = index_indices_by_rows(self.indices)

        if not self.dimensionality == 2:
            raise ValueError("Only 2d sparse arrays are supported")

    def transpose(self, *axes):
        """
        Transposes a sparse array. It just switches indices columns order.

        Returns
        -------
        SparseArray
            The transposed sparse array
        """
        _ax = np.array(axes)
        tp_indices = self.indices[:, _ax]
        return SparseArray(tp_indices, self.values, self.default_value, False)

    @property
    def T(self):
        return self.transpose(1, 0)

    def add(self, b):
        """
        Sparse arrays addition. Surprisingly,  did not manage to make __add__ work. See
        https://numba.readthedocs.io/en/stable/user/jitclass.html 'Supported operations'

        Parameters
        ----------
        b : SparseArray
            Another sparse array to add to self

        Returns
        -------
        SparseArray
            The resulting sparse array

        Raises
        ------
        ValueError
            If dimensionality is not 2 or len problem
        """
        a = self
        if a.dimensionality != b.dimensionality:
            raise ValueError("Dimensionality mismatch")

        # Pre-allocate arrays of indices and values for the resulting sparse
        idxs = np.empty((self.N + b.N, a.indices.shape[-1]), dtype=np.int64)
        values = np.empty((self.N + b.N,), dtype=np.float64)
        # Store result size
        counter = 0

        # navigate through a and b through these pointers
        apos = 0
        bpos = 0
        while (apos < self.N) & (bpos < b.N):
            # If A's first indices is smaller (upper-left), then add b first and increment B's pointer
            if (a.indices[apos, 0] > b.indices[bpos, 0]) or (
                (a.indices[apos, 0] == b.indices[bpos, 0])
                and (a.indices[apos, 1] > b.indices[bpos, 1])
            ):
                values[counter] = b.values[bpos]
                idxs[counter] = b.indices[bpos]
                counter += 1
                bpos += 1
            # The other way around
            elif (a.indices[apos, 0] < b.indices[bpos, 0]) or (
                (a.indices[apos, 0] == b.indices[bpos, 0])
                and (a.indices[apos, 1] < b.indices[bpos, 1])
            ):
                values[counter] = a.values[apos]
                idxs[counter] = a.indices[apos]
                counter += 1
                apos += 1
            # Else: indices are equal: then sum the elements and add it if necessary
            else:
                s = a.values[apos] + b.values[bpos]
                if s != self.default_value:
                    values[counter] = s
                    idxs[counter] = a.indices[apos]  # or b.indices[bpos], same here
                    counter += 1
                apos += 1
                bpos += 1

        # There might remain some elements in a:
        # Eg: a = [[0., 1.]]; b = [[1., 0.]]
        while apos < self.N:
            values[counter] = a.values[apos]
            idxs[counter] = a.indices[apos]
            counter += 1
            apos += 1
        # Same for b
        while bpos < self.N:
            values[counter] = b.values[bpos]
            idxs[counter] = b.indices[bpos]
            counter += 1
            bpos += 1

        return SparseArray(idxs[:counter], values[:counter], 0.0, True)

    def dense(self, shape):
        """
        Turn the sparse array into a dense matrix. Simply for testing purposes.

        NB: it is tricky to infer the shape based on indices max values due to numba tuple typing
        and array instanciation.

        Parameters
        ----------
        shape : tuple
            the final, desired shape

        Returns
        -------
        np.array
            Dense 2d array
        """
        target = np.full(shape, self.default_value)
        for idx, v in zip(self.indices, self.values):
            target[idx[0], idx[1]] = v

        return target

    def __getitem__(self, i):
        return self.values[self._row_indexer[i] : self._row_indexer[i + 1]]


@njit
def sort_indices(indices, values, dim=0):
    """
    Recursively sort indices by columns. This is rather fast ~ 30 ms for 10k 2d indices.

    Parameters
    ----------
    indices : np.array
        The indices array: (n_indices, dimensionality)
    values : np.array
        The values array, which needs to be sorted accordingly
    dim : int, optional
        The dimension to start sorting by, by default 0

    Returns
    -------
    Tuple
        Indices and values sorted accordingly
    """
    s = np.argsort(indices[:, dim])
    values = values[s]
    indices = indices[s]
    if dim < indices.shape[-1] - 1:
        rows = np.unique(indices[:, dim])
        for r in rows:
            sel = np.where(indices[:, dim] == r)[0]
            indices[sel], values[sel] = sort_indices(indices[sel], values[sel], dim + 1)
    elif dim == indices.shape[-1] - 1:
        pass

    return indices, values


@njit
def index_indices_by_rows(indices):
    """
    Generates an indexing array to facilitate access to values by rows:

    eg:
    ```python
    indices = np.array([[3, 0], [5, 0]])
    values = np.array([1., 2.])
    r = index_indices_by_rows(indices)
    # Get values at row 0:
    values[r[0]:r[0+1]] # Will return [] since no values on row 0
    # Get values at row 3
    values[r[3]:r[3+1]] # Will return [1.]
    ```

    Parameters
    ----------
    indices : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    bounds = [0]
    val = 0
    for r, i in enumerate(indices):
        while val < i[0]:
            bounds.append(r)
            val += 1
    bounds.append(r + 1)
    return np.array(bounds)
