from model.k_nearest_neighbor import KNearestNeighbor

import numpy as np

def test_fit():
    features = np.array([
        [1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4, 4],
        [5, 5, 5, 5, 5, 5],
    ])
    targets = np.array([1, 2, 1, 2, 1])
    model = KNearestNeighbor(1)
    model.fit(features, targets)

    assert model.features.shape[0] == 5
    assert model.targets.shape[0] == 5


def test_predict():
    features = np.array([
        [1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4, 4],
        [5, 5, 5, 5, 5, 5],
    ])
    targets = np.array([1, 2, 1, 2, 1])
    model = KNearestNeighbor(3)
    model.fit(features, targets)

    label = model.predict(np.array([1, 2, 3, 4, 5, 6]))

    assert label == 2
