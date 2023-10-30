from ..starter import knn

import pandas as pd

def test_knn():
    train_data_with_labels = pd.read_csv("../data/train.csv", header=None).values
    test_data_with_labels = pd.read_csv("../data/test.csv", header=None).values
    test_data_without_labels = test_data_with_labels[:, 1:785]

    predicted_labels = knn(train_data_with_labels, test_data_without_labels, "euclidean")

    assert len(predicted_labels) == test_data_with_labels.shape[0]

def test_kmeans():
    pass
