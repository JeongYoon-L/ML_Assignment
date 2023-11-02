import numpy as np
from ..utils.distance import euclidean

MAX_ITERATIONS = 100

class SoftKMeans:
    def __init__(self, n_clusters, beta):
        self.n_clusters = n_clusters
        self.distance_measure = "euclidean"
        self.beta = beta
        self.features = []
        self.labels = []
        self.means = []
        self.clusters = []

    def fit(self, features, labels):
        self.features = features
        self.labels = labels
        self.means = features[np.random.choice(len(features), size=self.n_clusters, replace=False)]

        for _ in range(MAX_ITERATIONS):
            distances = []

            for f in features:
                d = []
                for m in self.means:
                    d.append(euclidean(f, m))
                distances.append(d)

            distances = np.array(distances)
            clusters = np.exp(-1 * distances)
            clusters /= clusters.sum(axis=1)[:, np.newaxis]

            self.clusters = clusters
            self.means = distances.T.dot(features) / distances.sum(axis=0)[:, np.newaxis]

    def predict(self, features):
        if self.means is None:
            raise ValueError("Fit the model before making predictions")

        distances = []

        for m in self.means:
            distances.append(euclidean(features, m))

        distances = np.array(distances)

        prob_matrices = []

        for i, _ in enumerate(self.means):
            p = np.exp(-self.beta * distances[i]) / np.sum(np.exp(-self.beta * distances), axis=1, keepdims=True)
            prob_matrices.append(p)

        return prob_matrices
