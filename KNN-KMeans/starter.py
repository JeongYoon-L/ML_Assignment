from model.kmeans import KMeans
from model.k_nearest_neighbor import KNearestNeighbor

# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def knn(train, query, metric):
    labels = []
    features = []
    predicted_labels = []

    for t in train:
        labels.append(t[0] if isinstance(t[0], int) else int(t[0]))
        features.append(t[1])

    model = KNearestNeighbor(4, metric, "mode")
    model.fit(features, labels)

    for q in query:
        predicted_labels.append(model.predict(q[1]))

    return predicted_labels

# returns a list of labels for the query dataset based upon observations in the train dataset. 
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train, query, metric):
  if metric != 'euclidean' and metric != 'cosim':
    return (print("Please enter a valid metric"))

  if metric == 'euclidean':
    #euclidean knn
    kmeans_euclidean = KMeans(n_clusters=3, metric="euclidean")
    kmeans_euclidean.fit(train)
    predicted_labels = kmeans.predict(query)
    return (predicted_labels)

  if metric == 'cosim':
    kmeans_cosim = KMeans(n_clusters=3, metric="cosim")
    #cosine knn
    # return(labels)
    return None


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
    # show('valid.csv','pixels')
    # train_data = read_data('./train.csv')
    # valid_data = read_data('./valid.csv')
    # test_data = read_data('./test.csv')
    # print(kmeans(train_data, test_data, 'euclidean'))
    # train_data = pd.read_csv("./train.csv", header=None)
    # train_data = train_data.iloc[:, :785]
    # kmeans(train_data, test_data, 'euclidean')
    pass

    
if __name__ == "__main__":
    main()
    