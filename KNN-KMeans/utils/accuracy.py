from starter import read_data, knn

def get_knn_accuracy():
    train_data = read_data("../data/train.csv")
    test_data = read_data("../data/test.csv")

    predicted_labels = knn(train_data, test_data, "euclidean")

    score = 0

    for i, l in enumerate(predicted_labels):
        if l == test_data[i][0]:
            score += 1

    return score/len(test_data)


if __name__ == "__main__":
    print("knn accuracy on test/validation dataset", get_knn_accuracy())
