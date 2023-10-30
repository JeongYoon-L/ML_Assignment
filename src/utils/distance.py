import numpy as np

def euclidean(a, b):
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        return ValueError("The data type of two input vectors should be ndarray")

    if not a.shape == b.shape:
        return ValueError("The dimension of two input vectors should be same")

    return np.sqrt(np.sum((a - b) ** 2))

def cosim(a, b):
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        return ValueError("The data type of two input vectors should be ndarray")

    if not a.shape == b.shape:
        return ValueError("The dimension of two input vectors should be same")

    dot_prd = np.sum(a * b)
    mod_a = np.sqrt(np.sum(a ** 2))
    mod_b = np.sqrt(np.sum(b ** 2))

    return dot_prd / (mod_a * mod_b)
