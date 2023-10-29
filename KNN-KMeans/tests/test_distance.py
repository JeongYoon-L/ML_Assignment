from utils.distance import euclidean, cosim

def test_euclidean():
    a = [1, 2, 3, 4, 5, 6]
    b = [2, 3, 4, 5, 6, 7]

    assert euclidean(a, b) == 2.449489742783178


def test_cosim():
    a = [1, 2, 3, 4, 5, 6]
    b = [2, 3, 4, 5, 6, 7]

    assert cosim(a, b) == 0.9958408248979759
