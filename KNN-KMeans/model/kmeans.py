import numpy as np

class KMeans:
    def __init__(self, n_clusters, metric):
        """
        This class implements the traditional KMeans algorithm with hard assignments:

        https://en.wikipedia.org/wiki/K-means_clustering

        The KMeans algorithm has two steps:

        1. Update assignments
        2. Update the means

        While you only have to implement the fit and predict functions to pass the
        test cases, we recommend that you use an update_assignments function and an
        update_means function internally for the class.

        Use only numpy to implement this algorithm.

        Args:
            n_clusters (int): Number of clusters to cluster the given data into.

        """
        self.n_clusters = n_clusters
        self.means = None
        self.metric = metric

    def fit(self, features):
        """
        Fit KMeans to the given data using `self.n_clusters` number of clusters.
        Features can have greater than 2 dimensions.

        Args:
            features (np.ndarray):  array containing inputs of size
                (n_samples, n_features).
        Returns:
            None (saves model - means - internally)
        """
        centers = features[np.random.choice(len(features),size = self.n_clusters, replace = False)]
       
        new_centers = []
        labels = []


        if self.metric == 'euclidean':
            for _ in range(500): #set the random iter as 100
                
                for i, data_point in enumerate(features):
                    dist = [euclidean(data_point, mean) for mean in self.means]
                    closest_cluster = np.argmin(dist)
                    labels.append(closest_cluster)
                
                new_centers = np.array([features[labels == k].mean(axis=0) for k in range(self.n_clusters)])
                if(np.array_equal(centers, new_centers)):
                    break
                self.means = new_centers

        elif self.metric == 'cosine':
            for _ in range(500): #set the random iter as 100
                
                for i, data_point in enumerate(features):
                    dist = [cosim(data_point, mean) for mean in self.means]
                    closest_cluster = np.argmax(dist)
                    labels.append(closest_cluster)
                
                new_centers = np.array([features[labels == k].mean(axis=0) for k in range(self.n_clusters)])
                if(np.array_equal(centers, new_centers)):
                    break
                self.means = new_centers
            
                   

    def predict(self, features):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict cluster
        membership labels.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted cluster membership for each features,
                of size (n_samples,). Each element of the array is the index of the
                cluster the sample belongs to.
        """
        labels = []

        if self.metric == 'euclidean':
            for test_point in features :
                dist = [euclidean(test_point, mean) for mean in self.means]
                closest_cluster = np.argmin(dist)
                labels.append(closest_cluster)
                
        elif self.metric == 'cosine':
            for test_point in features:
                dist = [cosim(test_point, mean) for mean in self.means]
                closest_cluster = np.argmax(dist)
                labels.append(closest_cluster)

        return np.array(labels)
        
        raise NotImplementedError()
    

def euclidean(a,b):
  if len(a) != len(b):
    return ValueError("The dimenstion of two inpput vector should be same")

  e_dist = 0
  sum_of_dist = 0

  for i in range(len(a)):
    sum_of_dist += (a[i]-b[i]) ** 2
    
  e_dist = sum_of_dist ** 0.5

  return e_dist

def cosim(a,b):
  if len(a) != len(b):
    return ValueError("The dimenstion of two input vector should be same")

  dotProduct = 0

  for i in range(len(a)):
    dotProduct += a[i] * b[i]

  normA = (sum(x **2 for x in a)) ** 0.5
  normB = (sum(x **2 for x in b)) ** 0.5

  if normA ==0 or normB ==0:
    return 0

  c_dist = dotProduct / (normA * normB)

  return c_dist
