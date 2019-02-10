"""
Inspired by "A Tutorial on Spectral Clustering" by Ulrike von Luxburg
https://arxiv.org/abs/0711.0189
"""

from __future__ import division
import numpy as np
import scipy # for eigenvalues

from mlfromscratch.unsupervised_learning import KMeans
from mlfromscratch.supervised_learning import KNN


class SpectralClustering:
    """ An affinity based clustering method using the graph Laplacian

    Parameters:
    -----------
    n_clusters (int):
        The number of clusters to partition data
    normalization (str):
        The normalization method for the graph Laplacian. Can be None (unnormalized), "symmetric", or "random_walk"
    generalized_eigenproblem (bool):
        Whether to solve the generalized eigenproblem for the graph Laplacian as described by Shi and Malik (2000)
    renormalize (bool):
        Whether to normalize rows of the eigenvector matrix formed from the graph Laplacian
        as described in Ng, Jordan, and Weiss (2002). They suggest that suggest this method
        produces better clusterings which the degree to which different clusters are connected varies.
    """

    def __init__(self, n_clusters=2, affinity='rbf',
                 normalization='symmetric', generalized_eigenproblem=False, renormalize=True):

        self.labels_ = None
        self.affinity_matrix_ = None

        self._n_clusters = n_clusters
        self._affinity = affinity
        self._normalization = normalization
        self._generalized_eig = generalized_eigenproblem
        self._renormalize = renormalize

    def fit(self, X):
        if self._affinity == 'precomputed':
            self.affinity_matrix_ = X.copy()
        else:
            self.affinity_matrix_ = SpectralClustering.affinity_matrix(X, method=self._affinity)
        return self

    def predict(self):
        if self.affinity_matrix_ is None:
            raise ValueError("Must call fit() before predict()")

        L = self._laplacian()  # graph Laplacian

        if self._generalized_eig:
            D = np.diag(self.affinity_matrix_.sum(axis=1))
            w, v = scipy.linalg.eig(L, D)
        else:
            w, v = np.linalg.eig(L)

        order = w.argsort()
        U = v[:, order[:self._n_clusters]]  # first k eigenvalues

        if self._renormalize:
            row_sums = np.linalg.norm(U, axis=1)
            U = U / row_sums[:, np.newaxis]

        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(U[:,0], U[:,1])
        plt.show()

        kmeans = KMeans(k=self._n_clusters)
        self.labels_ = kmeans.predict(U)
        return self.labels_

    def fit_predict(self, X):
        self.fit(X)
        return self.predict()

    def _laplacian(self):
        m, _ = self.affinity_matrix_.shape

        W = self.affinity_matrix_.copy()
        d = W.sum(axis=1)
        D = None

        if self._normalization is None:
            # un-normalized graph Laplacian
            D = np.diag(d)
            L = D - W
        elif self._normalization == 'symmetric':
            D_sqrt_neg = np.diag(np.power(d, -0.5))
            I = np.eye(m)
            L = I - np.matmul(np.matmul(D_sqrt_neg, W), D_sqrt_neg)  # I - D^{-1/2} W D^{-1/2}
        elif self._normalization == 'random_walk':
            D_inv = np.diag(1 / d)
            I = np.eye(m)
            L = I - np.matmul(D_inv, W)  # I - D^{-1} W
        else:
            raise ValueError("Unrecognized Laplacian normalization: \'%s\'" % self._normalization)

        return L

    @staticmethod
    def affinity_matrix(X, method='rbf', dist='euclidian', **kwargs):
        if method == 'rbf':
            return SpectralClustering.rbf_affinity(X, dist=dist, **kwargs)

        elif method == 'eps_neighborhood':
            return SpectralClustering.epsilon_neighborhood_similarity(X, dist=dist, **kwargs)

        elif method == 'knn':
            return SpectralClustering.k_nearest_neighbor_graph(X, dist=dist, **kwargs)

        raise ValueError("Unknown similarity method \'%s\'" % method)

    @staticmethod
    def k_nearest_neighbor_graph(X, dist='euclidian', k=5, mutual=False):
        m, n = X.shape

        knn = KNN(k=k)
        indices = knn.kneighbors(X, X, return_distance=False)

        S = np.zeros((m, m), dtype=int)
        for i in range(m):
            S[i, indices[i]] += 1

        S = S + S.T
        if mutual:
            # mutual nearest neighbors
            S = np.where(S == 2, 1, 0)
        else:
            S = np.where(S > 0, 1, 0)

        return S

    @staticmethod
    def epsilon_neighborhood_similarity(X, dist='euclidian', epsilon=1.7):
        m, n = X.shape
        S = np.empty((m, m))
        for i in range(m):
            x = X[i]
            dists = np.linalg.norm(X - x, ord=2, axis=1)
            S[i, :] = (dists < epsilon).astype(float)
        return S

    @staticmethod
    def rbf_affinity(X, dist='euclidian', neighborhood_width=1):
        sigma = neighborhood_width
        m, n = X.shape

        S = np.empty((m, m))
        for i in range(m):
            xi = X[i]

            if dist == 'euclidian':
                dists = np.linalg.norm(X - xi, ord=2, axis=1)
                S[i, :] = np.exp(- dists / (2 * pow(sigma, 2)))

            elif dist == 'cosine':
                dists = 1 + np.matmul(X, xi) / (np.linalg.norm(xi) * np.linalg.norm(X, axis=1))
                dists = dists.real
                S[i, :] = np.power(dists, 2)

            else:
                raise ValueError("Unrecognized distance metric '%s'" % dist)
        return S

