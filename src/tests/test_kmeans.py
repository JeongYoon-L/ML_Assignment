from ..model.kmeans import KMeans

import numpy as np

def test_fit():
    features = np.array([
        [1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4, 4],
        [5, 5, 5, 5, 5, 5],
    ])

    model = KMeans(2)
    model.fit(features)

    assert len(model.means) == 2
    assert len(model.means[0]) == 6
