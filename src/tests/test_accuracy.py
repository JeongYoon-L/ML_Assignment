from ..utils.accuracy import get_accuracy

def test_get_accuracy():
    predicted_labels = [1, 2, 3, 4, 1]
    actual_labels = [1, 2, 3, 4, 5]

    assert get_accuracy(predicted_labels, actual_labels) == 0.8
