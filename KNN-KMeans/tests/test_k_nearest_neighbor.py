from model.k_nearest_neighbor import KNearestNeighbor

def test_fit():
    features = [
        [1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4, 4],
        [5, 5, 5, 5, 5, 5],
    ]
    targets = [1, 2, 1, 2, 1]
    model = KNearestNeighbor(1)
    model.fit(features, targets)

    assert len(model.features) == 5
    assert len(model.targets) == 5


def test_predict():
    features = [
        [1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4, 4],
        [5, 5, 5, 5, 5, 5],
    ]
    targets = [1, 2, 1, 2, 1]
    model = KNearestNeighbor(3)
    model.fit(features, targets)

    label = model.predict([1, 2, 3, 4, 5, 6])

    assert label == 2
