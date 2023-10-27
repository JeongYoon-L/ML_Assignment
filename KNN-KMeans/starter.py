# returns Euclidean distance between vectors a dn b
#Input : vectors a & b
#Ouput : scalar float value

# construct test cases and compare against results packages like numpy or sklearn

#Need to test!!!!!!!!!!!!!!!!!!!!!


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


# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def knn(train,query,metric):
    return(labels)

# returns a list of labels for the query dataset based upon observations in the train dataset. 
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train,query,metric):
    return(labels)

def read_data(file_name):
    
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
    return(data_set)
        
def show(file_name,mode):
    
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')
            
def main():
    show('valid.csv','pixels')
    
if __name__ == "__main__":
    main()
    