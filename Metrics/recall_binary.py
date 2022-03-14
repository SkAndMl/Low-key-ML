import numpy as np

def recall_binary(y_true,y_pred):
    """
    Calculates the recall score for a binary classifier using the formula TP/(TP+FN)
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: returns the recall score for the binary classifier
    """
    if len(y_true)!=len(y_pred):
        raise ValueError("Length mismatch: Lengths of y_true and y_pred don't match")

    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    positive = np.sum(y_true)
    true_positive = 0
    for i in range(len(y_pred)):
        if y_pred[i]==1 and y_pred[i]==y_true[i]:
            true_positive+=1

    return np.round(true_positive/positive,2)