from starter import knn, read_data

def test_knn():
    train_data = read_data("../data/train.csv")
    test_data = read_data("../data/test.csv")[:10]

    predicted_labels = knn(train_data, test_data, "euclidean")

    assert len(predicted_labels) == len([d[0] for d in test_data])


def test_kmeans():
    pass
