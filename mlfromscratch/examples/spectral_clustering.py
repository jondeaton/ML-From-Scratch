from sklearn import datasets

from mlfromscratch.utils import Plot
from mlfromscratch.unsupervised_learning import SpectralClustering


def main():

    # Load the dataset
    X, y = datasets.make_moons(n_samples=300, noise=0.08, shuffle=False)

    # affinity matrix
    # S = SpectralClustering.affinity_matrix(X, method='knn', dist='euclidian', k=5)

    # alternative affinities
    # S = SpectralClustering.affinity_matrix(X, method='eps_neighborhood', epsilon=1.7)
    S = SpectralClustering.affinity_matrix(X, method='rbf')

    n_clusters = 2

    # Un-normalized spectral clustering
    # sc = SpectralClustering(n_clusters, affinity='precomputed', normalization=None, renormalize=False)

    # Normalized spectral clustering according to Shi and Malik (2000)
    # sc = SpectralClustering(n_clusters, affinity='precomputed', normalization='symmetric', generalized_eigenproblem=True)

    # Normalized spectral clustering according to Ng, Jordan, and Weiss (2002)
    sc = SpectralClustering(n_clusters, affinity='precomputed', normalization='symmetric', renormalize=True)

    # predict classes
    y_pred = sc.fit_predict(S)

    # Project the data onto the 2 primary principal components
    p = Plot()
    p.plot_in_2d(X, y_pred, title="Spectral Clustering")
    p.plot_in_2d(X, y, title="Actual Clustering")


if __name__ == "__main__":
    main()
