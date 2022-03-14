import numpy as np

def precision_binary(y_true,y_pred):
    """
    Calculates the precision score for a binary classifier using the formula TP/(TP+FP)
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: returns the precision score for the binary classifier
    """
    if len(y_true)!=len(y_pred):
        raise ValueError("Length mismatch: Lengths of y_true and y_pred don't match")

    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    total_positive = np.sum(y_pred)
    true_positive = 0
    for i in range(len(y_pred)):
        if y_pred[i]==1 and y_pred[i]==y_true[i]:
            true_positive+=1

    return np.round(true_positive/total_positive,2)
