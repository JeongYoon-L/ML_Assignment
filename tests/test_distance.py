from ..utils.distance import euclidean, cosim

import numpy as np

def test_euclidean():
    a = np.array([1, 2, 3, 4, 5, 6])
    b = np.array([2, 3, 4, 5, 6, 7])

    assert euclidean(a, b) == 2.449489742783178


def test_cosim():
    a = np.array([1, 2, 3, 4, 5, 6])
    b = np.array([2, 3, 4, 5, 6, 7])

    assert cosim(a, b) == 0.9958408248979759
