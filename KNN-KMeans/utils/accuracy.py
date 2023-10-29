from starter import read_data, knn

import pandas as pd

def get_knn_accuracy():
    train_data_with_labels = pd.read_csv("../data/train.csv", header=None).values
    test_data_with_labels = pd.read_csv("../data/test.csv", header=None).values
    test_data_without_labels = test_data_with_labels[:, 1:785]

    predicted_labels = knn(train_data_with_labels, test_data_without_labels, "euclidean")

    score = 0

    for i, l in enumerate(predicted_labels):
        if l == test_data_with_labels[i, 0]:
            score += 1

    return score/test_data_with_labels.shape[0]


if __name__ == "__main__":
    print("knn accuracy on test/validation dataset", get_knn_accuracy())
