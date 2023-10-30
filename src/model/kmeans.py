from ..utils.distance import euclidean, cosim

import numpy as np
import math

MAX_ITERATIONS = 1000

class KMeans:
    def __init__(self, n_clusters, distance_measure='euclidean'):
        """
        This class implements the traditional KMeans algorithm with hard assignments:

        https://en.wikipedia.org/wiki/K-means_clustering

        The KMeans algorithm has two steps:

        1. Update assignments
        2. Update the means

        While you only have to implement the fit and predict functions to pass the
        test cases, we recommend that you use an update_assignments function and an
        update_means function internally for the class.

        Use only numpy to implement this algorithm.

        Args:
            n_clusters (int): Number of clusters to cluster the given data into.

        """
        self.n_clusters = n_clusters
        self.distance_measure = distance_measure
        self.features = []
        self.means = []

    def fit(self, features):
        """
        Fit KMeans to the given data using `self.n_clusters` number of clusters.
        Features can have greater than 2 dimensions.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            None (saves model - means - internally)
        """
        self.features = features
        self.means = features[np.random.choice(len(features), size=self.n_clusters, replace=False)]

        if self.distance_measure == "euclidean":
            distance_func = euclidean
        else:
            distance_func = cosim

        for i in range(MAX_ITERATIONS):
            clusters = {}

            for f in features:
                selected_mean_dist = math.inf
                selected_mean = None

                for m in self.means:
                    calculated_mean_dist = distance_func(f, m)

                    if calculated_mean_dist < selected_mean_dist:
                        selected_mean_dist = calculated_mean_dist
                        selected_mean = m

                selected_mean_hash = hash(selected_mean.tobytes())
                if selected_mean_hash not in clusters:
                    clusters[selected_mean_hash] = []

                clusters[selected_mean_hash].append(f)

            means = []
            for _, c in clusters.items():
                means.append(np.average(c, axis=0))

            self.means = means
            print("iteration %s, means %s, clusters %s" % (i, self.means, clusters))
            # print("iteration %s, means %s, clusters %s" % (i, len(self.means), 0))

    def predict(self, features):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict cluster
        membership labels.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted cluster membership for each features,
                of size (n_samples,). Each element of the array is the index of the
                cluster the sample belongs to.
        """
        raise NotImplementedError()
