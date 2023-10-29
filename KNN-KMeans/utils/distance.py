def euclidean(a, b):
    if len(a) != len(b):
        return ValueError("The dimension of two input vector should be same")

    e_dist = 0
    sum_of_dist = 0

    for i in range(len(a)):
        sum_of_dist += (a[i] - b[i]) ** 2

    e_dist = sum_of_dist ** 0.5

    return e_dist


def cosim(a, b):
    if len(a) != len(b):
        return ValueError("The dimension of two input vector should be same")

    dotProduct = 0

    for i in range(len(a)):
        dotProduct += a[i] * b[i]

    normA = (sum(x ** 2 for x in a)) ** 0.5
    normB = (sum(x ** 2 for x in b)) ** 0.5

    if normA == 0 or normB == 0:
        return 0

    c_dist = dotProduct / (normA * normB)

    return c_dist
