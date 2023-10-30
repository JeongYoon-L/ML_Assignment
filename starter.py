from src.model.k_nearest_neighbor import KNearestNeighbor
from src.model.kmeans import KMeans
from src.utils.accuracy import get_accuracy

# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def knn(train, query, metric):
    features = train[0]
    labels = train[1]

    model = KNearestNeighbor(4, metric, "mode")
    model.fit(features, labels)

    predicted_labels = []
    for q in query:
        predicted_labels.append(model.predict(q))

    return predicted_labels

# returns a list of labels for the query dataset based upon observations in the train dataset. 
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train, query, metric):
    features = train[0]
    labels = train[1]

    model = KMeans(4, metric)
    model.fit(features, labels)

    print(model.means)

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
            
def main():
    # knn
    train_data = read_data('.src/data/train.csv').values
    valid_data = read_data('.src/data/valid.csv').values
    test_data = read_data('.src/data/test.csv').values

    train = [train_data[:, 1:785], train_data[:, 0]]
    query = valid_data[:, 1:785]

    predicted_labels = knn(train, query, "euclidean")

    print("knn accuracy", get_accuracy(predicted_labels, valid_data[:, 0].tolist()))

    # kmeans
    kmeans(train, query, "euclidean")


if __name__ == "__main__":
    main()
    