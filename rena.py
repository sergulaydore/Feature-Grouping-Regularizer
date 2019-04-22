"""Recursive nearest agglomeration (ReNA):
    fastclustering for approximation of structured signals
"""
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.externals.joblib import Memory
from sklearn.externals import six
from scipy.sparse import csgraph, coo_matrix, dia_matrix
from sklearn.base import BaseEstimator
from sklearn.utils import check_array


def _compute_weights(masker, data_matrix):
    """Measuring the Euclidean distance: computer the weights in the direction
    of each axis

    Note: Here we are assuming a square lattice (no diagonal connections)
    """
    # data_graph shape
    dims = len(masker.mask_img_.shape)
    data_graph = masker.inverse_transform(data_matrix).get_data()
    weights = []

    for axis in range(dims):
        weights.append(
            np.sum(np.diff(data_graph, axis=axis) ** 2, axis=-1).ravel())

    return np.hstack(weights)


def _compute_edges(data_graph, is_mask=False):
    """
    """
    dims = len(data_graph.shape)
    edges = []
    for axis in range(dims):
        vertices_axis = np.swapaxes(data_graph, 0, axis)

        if is_mask:
            edges.append(np.logical_and(
                vertices_axis[:-1].swapaxes(axis, 0).ravel(),
                vertices_axis[1:].swapaxes(axis, 0).ravel()))
        else:
            edges.append(np.vstack(
                [vertices_axis[:-1].swapaxes(axis, 0).ravel(),
                 vertices_axis[1:].swapaxes(axis, 0).ravel()]))
    edges = np.hstack(edges)
    return edges


def _create_ordered_edges(masker, data_matrix):
    """
    """
    mask = masker.mask_img_.get_data()
    shape = mask.shape
    n_features = np.prod(shape)

    vertices = np.arange(n_features).reshape(shape)
    weights = _compute_weights(masker, data_matrix)
    edges = _compute_edges(vertices, is_mask=False)
    edges_mask = _compute_edges(mask, is_mask=True)

    # Apply the mask
    weights = weights[edges_mask]
    edges = edges[:, edges_mask]

    # Reorder the indices of the graph
    max_index = edges.max()
    order = np.searchsorted(np.unique(edges.ravel()), np.arange(max_index + 1))
    edges = order[edges]

    return edges, weights, edges_mask


def weighted_connectivity_graph(masker, data_matrix):
    """ Creating weighted graph

    data and topology, encoded by a connectivity matrix

    """
    n_features = masker.mask_img_.get_data().sum()

    edges, weight, edges_mask = _create_ordered_edges(masker, data_matrix)
    connectivity = coo_matrix(
        (weight, edges), (n_features, n_features)).tocsr()

    # Making it symmetrical
    connectivity = (connectivity + connectivity.T) / 2

    return connectivity


def _nn_connectivity(connectivity, thr):
    """ Fast implementation of nearest neighbor connectivity

    connectivity: weighted connectivity matrix
    """
    n_features = connectivity.shape[0]

    connectivity_ = coo_matrix(
        (1. / connectivity.data, connectivity.nonzero()),
        (n_features, n_features)).tocsr()

    inv_max = dia_matrix((1. / connectivity_.max(axis=0).toarray()[0], 0),
                         shape=(n_features, n_features))

    connectivity_ = inv_max * connectivity_

    # Dealing with eccentricities
    edge_mask = connectivity_.data > 1 - thr

    j_idx = connectivity_.nonzero()[1][edge_mask]
    i_idx = connectivity_.nonzero()[0][edge_mask]

    weight = np.ones_like(j_idx)
    edges = np.array((i_idx, j_idx))

    nn_connectivity = coo_matrix((weight, edges), (n_features, n_features))

    return nn_connectivity


def reduce_data_and_connectivity(labels, n_labels, connectivity, data_matrix,
                                 thr):
    """
    """
    n_features = len(labels)

    incidence = coo_matrix(
        (np.ones(n_features), (labels, np.arange(n_features))),
        shape=(n_labels, n_features), dtype=np.float32).tocsc()

    inv_sum_col = dia_matrix(
        (np.array(1. / incidence.sum(axis=1)).squeeze(), 0),
        shape=(n_labels, n_labels))

    incidence = inv_sum_col * incidence

    # reduced data
    reduced_data_matrix = (incidence * data_matrix.T).T
    reduced_connectivity = (incidence * connectivity) * incidence.T

    reduced_connectivity = reduced_connectivity - dia_matrix(
        (reduced_connectivity.diagonal(), 0), shape=(reduced_connectivity.shape))

    i_idx, j_idx = reduced_connectivity.nonzero()

    data_matrix_ = np.maximum(thr, np.sum(
        (reduced_data_matrix[:, i_idx] - reduced_data_matrix[:, j_idx]) ** 2, 0))
    reduced_connectivity.data = data_matrix_

    return reduced_connectivity, reduced_data_matrix


def nearest_neighbor_grouping(connectivity, data_matrix, n_clusters, thr):
    """ Cluster according to nn and reduce the data and connectivity
    """
    # Nearest neighbor conenctivity
    nn_connectivity = _nn_connectivity(connectivity, thr)

    n_features = connectivity.shape[0]

    n_labels = n_features - (nn_connectivity + nn_connectivity.T).nnz / 2

    if n_labels < n_clusters:
        # cut some links to achieve the desired number of clusters
        alpha = n_features - n_clusters

        nn_connectivity = nn_connectivity + nn_connectivity.T

        edges_ = np.array(nn_connectivity.nonzero())

        plop = edges_[0] - edges_[1]

        select = np.argsort(plop)[:alpha]

        nn_connectivity = coo_matrix(
            (np.ones(2 * alpha),
             np.hstack((edges_[:, select], edges_[::-1, select]))),
            (n_features, n_features))

    # Clustering step: getting the connected components of the nn matrix
    n_labels, labels = csgraph.connected_components(nn_connectivity)

    # Reduction step: reduction by averaging
    reduced_connectivity, reduced_data_matrix = reduce_data_and_connectivity(
        labels, n_labels, connectivity, data_matrix, thr)

    return reduced_connectivity, reduced_data_matrix, labels


def recursive_nearest_agglomeration(masker, data_matrix, n_clusters, n_iter,
                                    thr):
    """
    """
    # Weighted connectivity matrix
    connectivity = weighted_connectivity_graph(masker, data_matrix)

    # Initialization
    labels = np.arange(connectivity.shape[0])
    n_labels = connectivity.shape[0]

    for i in range(n_iter):
        connectivity, data_matrix, reduced_labels = nearest_neighbor_grouping(
            connectivity, data_matrix, n_clusters, thr)

        labels = reduced_labels[labels]
        n_labels = connectivity.shape[0]

        if n_labels <= n_clusters:
            break

    return n_labels, labels



class ReNA(BaseEstimator):
    """
    ReNA is useful.

    Parameters
    ----------
    masker: dd

    n_cluster: int, optional (default 2)
        Number of clusters.

    connectivity

    scaling: bool, optional (default False)

    memory: instance of joblib.Memory or string
        Used to cache the masking process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    n_iter: int, optional (default 10)
        Number of iterations of the recursive nearest agglomeration

    n_jobs: int, optional (default 1)
        Number of jobs in solving the sub-problems.

    thr: float in the opened interval (0., 1.), optional (default 1e-7)
        Threshold used to deal with eccentricities.

    Attributes
    ----------
    `labels_`: numpy array

    `n_clusters_`: int
        Number of clusters

    `sizes_`: numpy array
        It contains the size of each cluster

    """
    def __init__(self, n_clusters=2, connectivity=None, masker=None, memory=None,
                 scaling=False, n_iter=1000, thr=1e-7, n_jobs=1):
        self.n_clusters = n_clusters
        self.connectivity = connectivity
        self.memory = memory
        self.scaling = scaling
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.masker = masker
        self.thr = thr

    def fit(self, X):
        """Compute clustering of the data

        Parameters
        ----------
        X : 2D array
        """

        X = check_array(X, ensure_min_features=2)

        memory = self.memory
        if isinstance(memory, six.string_types):
            memory = Memory(cachedir=memory, verbose=0)

        if self.n_clusters <= 0:
            raise ValueError("n_clusters should be an integer greater than 0."
                             " %s was provided." % str(self.n_clusters))

        n_labels, labels = recursive_nearest_agglomeration(
            self.masker, X, self.n_clusters, n_iter=self.n_iter, thr=self.thr)

        sizes = np.bincount(labels)
        sizes = sizes[sizes > 0]

        self.labels_ = labels
        self.n_clusters_ = np.unique(self.labels_).shape[0]
        self.sizes_ = sizes
        self.n_features = X.shape[1]
        
        return self


    def transform(self, X):
        """Apply clustering, reduce the dimensionality of the data

        Parameters
        ----------
        X: 2D array
        """
        N = X.shape[0]
        check_is_fitted(self, 'labels_')
        Xred = np.array([np.bincount(self.labels_, X[i,:])/self.sizes_ for i in range(N)])

        if self.scaling:
            Xred = Xred * np.sqrt(self.sizes_)

        return Xred


    def fit_transform(self, X):
        """Fit to data, then perform the clustering (transformation)
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Xred):
        """
        """
        check_is_fitted(self, 'labels_')

        _, inverse = np.unique(self.labels_, return_inverse=True)

        if self.scaling:
            Xred = Xred / np.sqrt(self.sizes_)
        return Xred[..., inverse]
