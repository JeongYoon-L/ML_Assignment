from src.model.k_nearest_neighbor import KNearestNeighbor
from src.model.kmeans import KMeans
from src.model.soft_kmeans import SoftKMeans
from src.utils.accuracy import get_accuracy

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np

# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def knn(train, query, metric, aggregator, k_value):
    features = train[0]
    labels = train[1]

    query_features = query[0]

    model = KNearestNeighbor(k_value, metric, aggregator)
    model.fit(features, labels)

    predicted_labels = []
    for q in query_features:
        predicted_labels.append(model.predict(q))

    return predicted_labels


# returns a list of labels for the query dataset based upon observations in the train dataset. 
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train, query, metric):
    features = train[0]
    labels = train[1]

    query_features = query[0]

    model = KMeans(10, metric)
    model.fit(features, labels)

    # means = [c["mean"] for c in model.clusters]
    # plot_k_means_centroids(means)

    predicted_labels = []
    for q in query_features:
        predicted_labels.append(model.predict(q))
    len(predicted_labels)
    return predicted_labels


def soft_kmeans(train, query, beta):
    features = train[0]
    labels = train[1]

    query_features = query[0]

    print(features.shape)
    model = SoftKMeans(10, beta)
    model.fit(features, labels)

    # means = [c["mean"] for c in model.clusters]
    # plot_k_means_centroids(means)

    predicted_probs = []
    for q in query_features:
        predicted_probs.append(model.predict(q))
    len(predicted_probs)
    return predicted_probs


def read_data(file_name):
    
    data_set = []
    with open(file_name, 'rt') as f:
        for line in f:
            line = line.replace('\n', '')
            tokens = line.split(',')
            label = int(tokens[0])
            attribs = []
            for i in range(784):
                attribs.append(int(tokens[i+1]))
            data_set.append([label, attribs])

    return data_set


def show(file_name, mode):
    
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ', end='')
                else:
                    print('*', end='')
            else:
                print('%4s ' % data_set[obs][1][idx], end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0], end='')
        print(' ')


def get_best_k_value_for_knn(train, validation, metric, aggregator):
    train_features = train[0]
    train_labels = train[1]

    validation_features = validation[0]
    validation_labels = validation[1]

    K = 10
    best_k = 0
    best_acc = 0
    n_components = 60

    reduced_train_features, reduced_validation_features = get_reduced_features(
        train_features,
        validation_features,
        n_components
    )

    for k in range(1, K):
        model = KNearestNeighbor(k, metric, aggregator)
        model.fit(reduced_train_features, train_labels)

        predicted_labels = []
        for v in reduced_validation_features:
            predicted_labels.append(model.predict(v))

        acc = get_accuracy(predicted_labels, validation_labels.tolist())
        print("Knn validation accuracy for K = %d , PCA component = %d : %f" % (k, n_components, acc))

        if acc >= best_acc:
            best_acc = acc
            best_k = k

    return best_k


def plot_k_means_centroids(means):
    fig, axes = plt.subplots(2, 5, figsize=(8, 3))

    for i, ax in enumerate(axes.flat):
        if i < len(means):
            ax.imshow(means[i].reshape(28, 28), cmap="gray")
            ax.set_title(f"Cluster {i}")
        ax.axis("off")

    plt.show()

def create_confusion_matrix(true_labels, predicted_labels):
    confusion_matrix = np.zeros((10, 10), dtype=int)

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        confusion_matrix[int(true_label)][int(predicted_label)] += 1

    return confusion_matrix


def plot_confusion_matrix(confusion_matrix, class_names, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap="Blues", linewidths=0.5, linecolor="black")
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title(title)
    plt.show()


def get_reduced_features(train_features, test_features, n_components):
    pca = PCA(n_components)
    reduced_train_features = pca.fit_transform(train_features)
    reduced_test_features = pca.transform(test_features)
    return reduced_train_features, reduced_test_features

def main():
    # knn
    train_data = pd.read_csv("./src/data/train.csv", header=None).values
    test_data = pd.read_csv("./src/data/test.csv", header=None).values
    validation_data = pd.read_csv("./src/data/valid.csv", header=None).values

    train = [train_data[:, 1:785], train_data[:, 0]]
    test = [test_data[:, 1:785], test_data[:, 0]]
    validation = [validation_data[:, 1:785], validation_data[:, 0]]

    best_k_value = get_best_k_value_for_knn(
        train,
        validation,
        "euclidean",
        "mode"
    )
    reduced_train_features, reduced_test_features = get_reduced_features(
        train[0],
        test[0],
        60
    )
    best_k_value_predicted_labels = knn(
        [reduced_train_features, train[1]],
        [reduced_test_features],
        "euclidean",
        "mode",
        best_k_value,
    )
    best_k_value_accuracy = get_accuracy(
        best_k_value_predicted_labels,
        test[1].tolist()
    )
    print("Knn test accuracy for K = %d : %f" % (best_k_value, best_k_value_accuracy))

    #confusion matrix
    class_names = [str(i) for i in range(10)]
    confusion_matrix_knn = create_confusion_matrix(test[1].tolist(), best_k_value_predicted_labels)
    plot_confusion_matrix(confusion_matrix_knn, class_names, "K-Nearest Neighbors Confusion Matrix")


    # kmeans
    predicted_labels = kmeans(train, validation, "euclidean")
    print("K-Means validation accuracy : %f" % (get_accuracy(predicted_labels, validation[1].tolist())))

    predicted_labels = kmeans(train, test, "euclidean")
    print("K-Means test accuracy : %f" % (get_accuracy(predicted_labels, test[1].tolist())))

    #confusion matrix
    confusion_matrix_kmeans = create_confusion_matrix(test[1].tolist(), predicted_labels)
    plot_confusion_matrix(confusion_matrix_kmeans, class_names, "K-Means Confusion Matrix")


    # soft kmeans
    # soft_kmeans(train, validation, 0.5)


if __name__ == "__main__":
    main()
    