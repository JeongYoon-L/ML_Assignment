from src.model.k_nearest_neighbor import KNearestNeighbor
from src.model.kmeans import KMeans
from src.utils.accuracy import get_accuracy

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
    # labels = train[1]

    model = KMeans(9, metric)
    model.fit(features)

    # print(model.means)

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

    pca = PCA(n_components)
    reduced_train_features = pca.fit_transform(train_features)
    reduced_validation_features = pca.transform(validation_features)

    for k in range(1, K):
        model = KNearestNeighbor(k, metric, aggregator)
        model.fit(reduced_train_features, train_labels)

        predicted_labels = []
        for v in reduced_validation_features:
            predicted_labels.append(model.predict(v))

        acc = get_accuracy(predicted_labels, validation_labels.tolist())
        print("knn test accuracy for K = %d , PCA = %d : %f" % (k, n_components, acc))

        if acc > best_acc:
            best_acc = acc
            best_k = k

    return best_k


def main():
    # knn
    train_data = pd.read_csv('./src/data/train.csv').values
    test_data = pd.read_csv('./src/data/test.csv').values
    validation_data = pd.read_csv('./src/data/valid.csv').values

    train = [train_data[:, 1:785], train_data[:, 0]]
    test = [test_data[:, 1:785], test_data[:, 0]]
    validation = [validation_data[:, 1:785], validation_data[:, 0]]

    best_k_value = get_best_k_value_for_knn(train, validation, "euclidean", "mode")
    best_k_value_predicted_labels = knn(train, test, "euclidean", "mode", best_k_value)
    best_k_value_accuracy = get_accuracy(best_k_value_predicted_labels, test[1].tolist())

    print("knn validation accuracy for K = %d : %f" % (best_k_value, best_k_value_accuracy))

    # kmeans
    # kmeans(train, query, "euclidean")


if __name__ == "__main__":
    main()
    