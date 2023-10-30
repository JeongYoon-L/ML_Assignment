def get_accuracy(predicted_labels, actual_labels):
    score = 0

    for i, l in enumerate(predicted_labels):
        if l == actual_labels[i]:
            score += 1

    return score/len(actual_labels)
