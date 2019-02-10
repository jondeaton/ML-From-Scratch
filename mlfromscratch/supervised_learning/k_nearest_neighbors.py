from __future__ import print_function, division
import numpy as np
from mlfromscratch.utils import euclidean_distance

class KNN():
    """ K Nearest Neighbors classifier.

    Parameters:
    -----------
    k: int
        The number of closest neighbors that will determine the class of the 
        sample that we wish to predict.
    """
    def __init__(self, k=5):
        self.k = k

    def _vote(self, neighbor_labels):
        """ Return the most common class among the neighbor samples """
        counts = np.bincount(neighbor_labels.astype('int'))
        return counts.argmax()

    def predict(self, X_test, X_train, y_train):
        indices = self.kneighbors(X_test, X_train, return_distance=False)
        y_pred = np.empty(X_test.shape[0])
        # Determine the class of each sample
        for i, idx in enumerate(indices):
            # Extract the labels of the K nearest neighboring training samples
            k_nearest_neighbors = np.array([y_train[i] for i in idx])
            # Label sample as the most common class label
            y_pred[i] = self._vote(k_nearest_neighbors)

        return y_pred

    def kneighbors(self, X_test, X_train, return_distance=True):
        indices = np.empty((X_test.shape[0], self.k), dtype=int)
        if return_distance:
            distances = np.empty((X_test.shape[0], self.k))

        for i, test_sample in enumerate(X_test):
            # Sort the training samples by their distance to the test sample and get the K nearest
            dists = [euclidean_distance(test_sample, x) for x in X_train]
            indices[i] = np.argsort(dists)[:self.k]
            if return_distance:
                distances[i] = sorted(dists)[:self.k]

        if return_distance:
            return distances, indices
        else:
            return indices
