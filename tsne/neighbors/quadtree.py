import os
from collections import OrderedDict

import numpy as np
from numba import (boolean, deferred_type, float64, int64, njit, optional,
                   prange, typed, types)
from numba.experimental import jitclass
from tsne.neighbors.utils import (current_furthest, distance_to_hyperplan,
                                  englobing_box, heap_push, heap_push_pop,
                                  heap_sort, init_heap, max_spread_axis,
                                  quadsplit, rdist, split_along_axis)

base_node_specs = OrderedDict()
node_type = deferred_type()
base_node_specs["data"] = float64[:, :]
base_node_specs["bbox"] = optional(float64[:])
base_node_specs["centroid"] = optional(float64[:])
base_node_specs["diagonal"] = optional(float64)
# base_node_specs["axis"] = int64
base_node_specs["dimensionality"] = int64
base_node_specs["indices"] = optional(int64[:])
base_node_specs["is_leaf"] = boolean
base_node_specs["leaf_size"] = int64
base_node_specs["N"] = int64
# base_node_specs["left"] = optional(node_type)
# base_node_specs["right"] = optional(node_type)
base_node_specs["nw"] = optional(node_type)
base_node_specs["ne"] = optional(node_type)
base_node_specs["sw"] = optional(node_type)
base_node_specs["se"] = optional(node_type)
# base_node_specs['children'] = types.ListType(node_type)
# base_node_specs['children'] = optional(node_type)


@jitclass(base_node_specs)
class Node:
    """
    Main object for the node class.

    Note that this implementation is suboptimal: it manages to match sklearn building process
    only because leaf size is set to 16 or above. Slowdown is due to the bulkloading process,
    which is utterly sub-optimal for quadtrees

    Note that the tree building process is peculiar:
    Since jit classes methods can not be recursive (node.method can not call node.method), the
    tree building process (recursive node splitting and children instanciation) can not be done
    inside the Node.__init__ method (which is the natural way to do so).
    However, jit classes methods can call recursive functions: hence, the tree building process is
    delegated to an independant function (see `build` function).
    Consequently, this class must be used in the following way:
    ```
    # Instanciate the root node
    node = Node(data)
    # Reccursively attach children to each node
    node.build()
    ```

    For convenience, a wrapper `KDTree` class was implemented, encapsulating this process:
    ```
    tree = KDTree(data)
    tree.query_radius(...)
    ```
    """

    def __init__(self, data, leaf_size=16, indices=None):
        # Stores the data
        self.data = data
        self.N = len(data)
        self.dimensionality = data.shape[-1]

        if self.dimensionality != 2:
            raise ValueError(
                "QuadTree only supports 2-d data, see https://github.com/numba/numba/issues/6916"
            )

        # This is actually possible in quadtrees
        # if len(self.data) == 0:
        #     raise ValueError("Empty data")

        # Stores indices of each data point
        if indices is None:
            self.indices = np.arange(len(data))
        else:
            self.indices = indices

        self.leaf_size = leaf_size

        # Is it a leaf
        if len(data) <= leaf_size:
            self.is_leaf = True
        else:
            self.is_leaf = False

        # Determine node bounding box
        if len(data) > 0:
            self.bbox = englobing_box(self.data)
            self.centroid = np.array([self.data[:, 0].mean(), self.data[:, 1].mean()])
            self.diagonal = np.power(self.bbox[2:] - self.bbox[:2], 2.0).sum()
        else:
            self.bbox = None
            self.centroid = None
            self.diagonal = None

        # Pre-assign empty children for typing
        # self.children = None
        self.nw = None
        self.ne = None
        self.sw = None
        self.se = None
        # self.children = typed.List.empty_list(node_type)

    def __repr__(self):
        return f"{'Leaf' if self.is_leaf else ''}Node containing {self.N} points inside {self.bbox} with centroid {self.centroid}"

    def split(self):
        """
        Splits a node into two children nodes.

        Returns
        -------
        Tuple[Node]
            Left children and right children
        """
        indices = quadsplit(self.data)
        children = []
        for idxs in indices:
            # Simply reference the data in the children, do not copy arrays
            child = Node(self.data[idxs], self.leaf_size, self.indices[idxs])
            children.append(child)

        return children

    def split_and_assign(self):
        """
        Assigns the children node.
        Strangely enough, this needs to be delegated to an explicit method.
        """
        nw, ne, sw, se = self.split()
        self.nw = nw
        self.ne = ne
        self.sw = sw
        self.se = se

    def contains_point(self, point):
        """Whether the node contains or not this point.
        It actually checks if the point is inside the node bbox

        Parameters
        ----------
        point : np.array
            k-d Point to check

        Returns
        -------
        bool
            True or False
        """
        return (point >= self.bbox[:2]).all() & (point <= self.bbox[2:]).all()

    def build(self):
        """
        Reccursively build the children.
        Jit methods can not be explicitely recursive:
        `self.build` can not call `self.build`, but it can call a function
        which calls itself, the workaround used here.
        """
        # Reccursively attach children to the parent
        build(self, True)

    def query_radius(self, X, max_radius):
        """
        Return set of point in the dataset distance from less than `radius` to the query point X.

        Parameters
        ----------
        X : np.array
            Query point (single point only)
        radius : float
            max radius

        Returns
        -------
        np.array
            Indices of points within that radius from the query point

        Raises
        ------
        ValueError
            This function works on single query point only.
        """
        if X.ndim > 1:
            raise ValueError("query_radius only works on single query point.")
        if X.shape[-1] != self.dimensionality:
            raise ValueError("Tree and query dimensionality do not match")
        # Initialize empty list of int64
        # Needs to be typed
        buffer = [0][:0]

        # Query recursive
        # Fills in-place the neighbors list
        query_radius(self, X, max_radius ** 2, buffer)

        # return np array for convenience
        return np.array(buffer)

    def query_nearest(self, X, k=1):
        """
        Nearest neighbor search in the QuadTree. This is actually not used in this repo.

        Parameters
        ----------
        X : np.array
            input point
        k : int, optional
            Number of nearest neighbor to limit search to, by default 1

        Returns
        -------
        Tuple
            Square distances, indices

        Raises
        ------
        ValueError
            If dimensionalities mismatch
        ValueError
            If several points are queried simultaneously
        """
        if X.ndim > 1:
            raise ValueError("query_nearest only works on single query point.")
        if X.shape[-1] != self.dimensionality:
            raise ValueError("Tree and query dimensionality do not match")

        heap = init_heap()
        query_nearest(self, X, k, heap, 0.0, True)

        # If there are not k points in the tree
        for _ in range(len(heap) - k):
            heap_push(heap, 1.0e9, -1)

        distances, indices = heap_sort(heap)

        return distances, indices

    def query_many_nearest(self, X, k=1):
        """
        Wraper to run knn search in parallel.

        Parameters
        ----------
        X : np.array
            Input points to query for (n_queries, dimensionality)
        k : int, optional
            Amount of knn to limit search to, by default 1

        Returns
        -------
        Tuple
            Square distances, indices, both of shape (n_queries, k)

        Raises
        ------
        ValueError
            If dimensionalities mismatch
        ValueError
            If several points are queried simultaneously
        """
        if X.ndim != 2:
            raise ValueError("query_many_nearest only works on multi query point.")

        if X.shape[-1] != self.dimensionality:
            raise ValueError("Tree and query dimensionality do not match")

        distances, indices = query_many_nearest(self, X, k)

        return distances, indices

    def order_children(self, point):
        """
        Utilty to order children by proximity to the given point.
        Used in knn search.

        Parameters
        ----------
        point : np.array
            Input point

        Returns
        -------
        Tuple
            List of ordered children, List of corresponding distances to point

        Raises
        ------
        ValueError
            If leaf
        """
        if self.is_leaf:
            raise ValueError("Leaf has no children")

        distances = []
        children = [self.nw, self.ne, self.sw, self.se]
        for child in children:
            if child.bbox is not None:
                dist = distance_to_hyperplan(point, child.bbox)
            else:
                dist = 1.0e9
            distances.append(dist)

        order = np.argsort(np.array(distances))

        ordered_children = [children[i] for i in order]
        ordered_distances = [distances[i] for i in order]
        return ordered_children, ordered_distances

    def distance_to_point(self, point):
        """
        Distance from node centroid to point.
        Used in Barnes Hut resultant forces computations.

        Parameters
        ----------
        point : np.array
            Input point

        Returns
        -------
        Float
            Square distance to point
        """
        return rdist(self.centroid, point)

    def compute_resultant(self, i, theta, F, Z):
        """
        Compute resultant using the Barnes Hut approximation.
        F and Z are muted in-place. For this reason, this function is not to be used as is,
        but rather to be wrappe by its parallel multi-query counterpart.

        Parameters
        ----------
        i : Integer
            Index of the point onto which to compute resultant forces
        theta : Float
            Angle
        F : np.array
            Resulting forces placeholder
        Z : np.array
            Resulting Z (sum of joint student probability density in the embedded space)
        """

        # F and Z are to be muted (modified in place)
        bh_resultant(self, self.data[i], i, theta, Z, F)

    def compute_many_resultants(self, theta):
        """
        Simple wrapper for multi-barnes hut approximation.
        It computes resultant forces of all the points indexed in the root node.

        Parameters
        ----------
        theta : float
            Limit angle below which a cell is aggregated.

        Returns
        -------
        Tuple
            F, the resulting forces. Z, (sum of joint student probability density in the embedded space)
        """
        return compute_many_resultants(self, theta)


@njit(parallel=True)
def compute_many_resultants(root, theta):
    """
    Since jitclass method can not (as of now) be annotated with the 'parallel' keyword,
    this has to be delegated in a proper function.

    Parameters
    ----------
    root : Node
        QuadTree root noe
    theta : Float
        Limit angle

    Returns
    -------
    Tuple
        F, the resulting forces. Z, (sum of joint student probability density in the embedded space)
    """

    F = np.zeros_like(root.data)
    # This needs to be mutable, ie of non-zero size, even its slices
    Z = np.zeros((len(root.data), 1), dtype=np.float64)
    # Z = np.array(0., dtype=np.float64)
    for i in prange(root.N):
        root.compute_resultant(i, theta, F[i], Z[i])

        # if np.isnan(F[i]).any():
        #     import ipdb; ipdb.set_trace()

        # Z += z
        # F[i] = repulsion
    # import ipdb; ipdb.set_trace()
    Z = Z.sum()
    # print('Z ->', Z)
    return F, Z


@njit
def bh_resultant(node, point, i, theta, Z, F):
    """
    Recursive barnes hut approximation resulting forces computation.

    Parameters
    ----------
    node : Node
        Considered node
    point : np.array
        Point on to which resulting forces are computed
    i : Integer
        Index of this point
    theta : Float
        Limit angle
    Z : np.array
        Placeholder for resulting forces computations
    F : np.array
        Placeholder for Z computation
    """
    if node is None:
        return

    if len(node.data) == 0:
        return

    if len(node.data) == 1:
        d = rdist(point, node.data[0])
        d += 1
        d **= -1
        Z += d
        F += d * (point - node.data[0])
        return

    d_cell = node.distance_to_point(point)
    angle = node.diagonal / d_cell
    if angle < theta:
        Ncell = node.N

        if node.contains_point(point):
            Ncell -= 1

        q_i_cell = 1.0 / (1.0 + d_cell)
        weight = q_i_cell * Ncell
        Z += weight
        for k in range(node.dimensionality):
            F[k] += weight * (point[k] - node.centroid[k]) * q_i_cell

        # Break the search
        # print('Summarizing ', node.N, 'points', node.bbox)
        return

    elif node.is_leaf:
        # print('Leaf, angle', angle)
        for j, ref in zip(node.indices, node.data):
            if i == j:
                continue
            q_i_j = 1 / (1.0 + rdist(ref, point))
            Z += q_i_j
            # !!! This call blocks proper parallelization if vectorized, see https://github.com/numba/numba/issues/2699
            for k in range(node.dimensionality):
                F[k] += (point[k] - ref[k]) * q_i_j * q_i_j

    else:
        # print('Non leaf, angle', angle, node.N, 'points', 'at', node.bbox)
        for subnode in [node.nw, node.ne, node.sw, node.se]:
            bh_resultant(subnode, point, i, theta, Z, F)


if os.environ.get("NUMBA_DISABLE_JIT", "0") != "1":
    node_type.define(Node.class_type.instance_type)


@njit
def build(current, is_root=False):
    """
    Reccursive building process.
    Since jit methods can not be recursive, it has to be a detached function.
    Otherwise, it would just be included inside the Node.__init__ method.

    Parameters
    ----------
    current : Node
        Current node to split if needed
    """
    # print('Building', len(current.data), current.is_leaf)
    if not current.is_leaf:
        current.split_and_assign()
        # for child in current.children:
        #     build(child)
        build(current.nw, False)
        build(current.ne, False)
        build(current.sw, False)
        build(current.se, False)
        # build(current.children)

    # if is_root:
    #     import ipdb; ipdb.set_trace()


@njit
def query_radius(node, X, max_radius, buffer, dist_LB=0.0, is_root=True):
    """
    This function should not be used as-is: jitted-class methods can not be recursive.
    The recursive query process is delegated here.

    This is a depth-first search: by ensuring that one chooses the closest
    node first, the algorithm will consequently first go to the node containing the point (if any)
    and then go backward in neighbors node, trimming each node too far from the `max_radius` given.

    Parameters
    ----------
    node: Node
        Currently visited node
    X : np.array
        Query point (one point).
    max_radius : float
        Max radius
    buffer : list
        List of currently-gathered neighbors. Stores in-place the neighbors along the search process
    dist_LB : float, optional
        Distance lower bound: distance from the query point to the currently visited node's
        hyperplan, by default 0.0
    is_root : bool, optional
        Whether the currently visited node is root, by default True
    """
    if is_root:
        # If first call, no lower bound distance has already been computed
        dist_LB = distance_to_hyperplan(X, node.bbox)

    # if query is outside the radius, then trim this node out
    if dist_LB > max_radius:
        return

    # If it's a leaf: check points inside
    elif node.is_leaf:
        for i, y in zip(node.indices, node.data):
            d = rdist(X, y)
            if d <= max_radius:
                buffer.append(i)

    # Else, continue search
    # Going for the closest node first ensures a depth-first search
    else:
        left_LB = distance_to_hyperplan(X, node.left.bbox)
        right_LB = distance_to_hyperplan(X, node.right.bbox)

        if left_LB < right_LB:
            query_radius(node.left, X, max_radius, buffer, left_LB, False)
            query_radius(node.right, X, max_radius, buffer, right_LB, False)
        else:
            query_radius(node.right, X, max_radius, buffer, right_LB, False)
            query_radius(node.left, X, max_radius, buffer, left_LB, False)


@njit
def query_nearest(node, X, k, heap, dist_LB=0.0, is_root=True):
    """
    This function should not be used as-is: jitted-class methods can not be recursive.
    The recursive query process is delegated here.

    This is a depth-first search: by ensuring that one chooses the closest
    node first, the algorithm will consequently first go to the node containing the point (if any)
    and then go backward in neighbors node, trimming each node too far from the `max_radius` given.


    Parameters
    ----------
    node : Node
        Considered node
    X : np.array
        Point wich neighbors are queried
    k : integer
        Amount of nearest neighbors to limit search to
    heap : List
        Running heapqueue for knn storage
    dist_LB : float, optional
        Lower bound distance between the point and the node, by default 0.0
    is_root : bool, optional
        Is the node root, by default True
    """
    # Keep a heapqueue running in order to store neighbors
    # We know that first element in the heap is the smallest, ie the biggest inverse distance
    if is_root:
        # If first call, no lower bound distance has already been computed
        dist_LB = distance_to_hyperplan(X, node.bbox)

    # if the this node is further than the already k-th furthest neighbor, trim this node
    if dist_LB > current_furthest(heap) and len(heap) == k:
        pass

    # If it's a leaf: check points inside
    elif node.is_leaf:
        for i, y in zip(node.indices, node.data):
            d = rdist(X, y)
            if len(heap) < k:
                heap_push(heap, d, i)
            elif d < -heap[0][0]:
                heap_push_pop(heap, d, i)

    # Else, continue search
    # Going for the closest node first ensures a depth-first search
    else:
        # left_LB = distance_to_hyperplan(X, node.left.bbox)
        # right_LB = distance_to_hyperplan(X, node.right.bbox)
        children, distances = node.order_children(X)
        for child, dist_LB in zip(children, distances):
            query_nearest(child, X, k, heap, dist_LB, False)


@njit(parallel=True)
def query_many_nearest(node, X, k=1):
    """
    Simple wrapper to annotate `parallel=True` the knn search.

    Parameters
    ----------
    node : Node
        Root node
    X : np.array
        Input points to query knn on: (n_queries, dimensionality)
    k : int, optional
        Amount of nearest neighbors to limit search to, by default 1

    Returns
    -------
    Tuple
        Distances and indices
    """
    distances = np.empty((len(X), k), dtype=np.float64)
    indices = np.empty((len(X), k), dtype=np.int64)
    for i in prange(len(X)):
        d, idxs = node.query_nearest(X[i], k)
        distances[i] = d
        indices[i] = idxs
    return distances, indices


specs = OrderedDict()
specs["_root"] = node_type
specs["data"] = float64[:, :]


@jitclass(specs)
class QuadTree:
    """
    Simple wrapper class to build the tree, it encapsulates root node instanciation,
    followed by the recursive building process, which can not be called in node.__init__
    because jitted methods can not be recursive.

    """

    def __init__(self, data, leaf_size=16):
        self.data = data
        self._root = Node(data, leaf_size, None)
        self._root.build()

    def query_radius(self, X, radius):
        """
        Return set of point in the dataset distance from less than `radius` to the query point X.

        Parameters
        ----------
        X : np.array
            Query point (single point only)
        radius : float
            max radius

        Returns
        -------
        np.array
            Indices of points within that radius from the query point
        """
        return self._root.query_radius(X, radius)

    def query_nearest(self, X, k=1):
        """
        Nearest neighbor search in the QuadTree. This is actually not used in this repo.

        Parameters
        ----------
        X : np.array
            input point
        k : int, optional
            Number of nearest neighbor to limit search to, by default 1

        Returns
        -------
        Tuple
            Square distances, indices

        Raises
        ------
        ValueError
            If dimensionalities mismatch
        ValueError
            If several points are queried simultaneously
        """
        return self._root.query_nearest(X, k)

    def query_many_nearest(self, X, k=1):
        """
        Wraper to run knn search in parallel.

        Parameters
        ----------
        X : np.array
            Input points to query for (n_queries, dimensionality)
        k : int, optional
            Amount of knn to limit search to, by default 1

        Returns
        -------
        Tuple
            Square distances, indices, both of shape (n_queries, k)

        Raises
        ------
        ValueError
            If dimensionalities mismatch
        ValueError
            If several points are queried simultaneously
        """
        return self._root.query_many_nearest(X, k)

    def compute_resultant(self, i, F, Z, theta=1.0):
        """
        Compute resultant using the Barnes Hut approximation.
        F and Z are muted in-place. For this reason, this function is not to be used as is,
        but rather to be wrappe by its parallel multi-query counterpart.

        Parameters
        ----------
        i : Integer
            Index of the point onto which to compute resultant forces
        theta : Float
            Angle
        F : np.array
            Resulting forces placeholder
        Z : np.array
            Resulting Z (sum of joint student probability density in the embedded space)
        """
        return self._root.compute_resultant(X, theta)

    def compute_many_resultants(self, theta=1.0):
        """
        Simple wrapper for multi-barnes hut approximation.
        It computes resultant forces of all the points indexed in the root node.

        Parameters
        ----------
        theta : float
            Limit angle below which a cell is aggregated.

        Returns
        -------
        Tuple
            F, the resulting forces. Z, (sum of joint student probability density in the embedded space)
        """
        return self._root.compute_many_resultants(theta)
