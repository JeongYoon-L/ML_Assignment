import numpy as np
from ..utils.distance import euclidean

from statistics import mode
from math import sqrt
from collections import Counter
import numpy as np
import pandas as pd
import math


class SoftKMeans:
    def __init__(self, n_neighbors, distance_measure='euclidean', aggregator='mode'):
        self.n_neighbors = n_neighbors
        self.distance_measure = distance_measure
        self.aggregator = aggregator
        self.centroids = None

    def _calculate_distances(self, features):
        distances = []
        for centroid in self.centroids:
            if features.ndim == 1:
                if self.distance_measure == 'euclidean':
                    dist = np.linalg.norm(features - centroid)
                elif self.distance_measure == 'cosim':
                    dist = np.dot(features, centroid) / (np.linalg.norm(features) * np.linalg.norm(centroid))
                    dist = 1 - dist  # Convert cosine similarity to distance
            else:
                if self.distance_measure == 'euclidean':
                    dist = np.linalg.norm(features - centroid, axis=1)
                elif self.distance_measure == 'cosim':
                    dist = np.dot(features, centroid) / (np.linalg.norm(features) * np.linalg.norm(centroid))
                    dist = 1 - dist  # Convert cosine similarity to distance
            distances.append(dist)
        return np.array(distances)

    def fit(self, features, targets):
        num_clusters = len(set(targets))
        self.centroids = np.zeros((num_clusters, features.shape[1]))
        
        for cluster_id in range(num_clusters):
            cluster_features = features[targets == cluster_id]
            if len(cluster_features) == 0:
                self.centroids[cluster_id] = features.mean(axis=0)
            else:
                self.centroids[cluster_id] = cluster_features.mean(axis=0)


    def predict(self, features, ignore_first=False):
        if self.centroids is None:
            raise ValueError("Fit the model before making predictions")

        distances = self._calculate_distances(features)
        if ignore_first:
            distances = distances[:, 1:]  # Ignore the first nearest centroid

        prob_matrices = []
        for centroid in self.centroids:
            if features.ndim == 1:
                prob_matrix = np.exp(-distances) / np.sum(np.exp(-distances))
            else:
                prob_matrix = np.exp(-distances) / np.sum(np.exp(-distances), axis=1, keepdims=True)
            prob_matrices.append(prob_matrix)
        return prob_matrices




