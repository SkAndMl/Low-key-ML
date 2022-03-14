import numpy as np

def accuracy_score(y_true,y_pred):
    """
    Returns the accuracy score of a binary classifier
    :param y_true: true label
    :param y_pred: predicted label
    :return: accuracy score
    """
    if len(y_true)!=len(y_pred):
        raise ValueError("Length mismatch: Lengths of y_true and y_pred don't match")
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    return np.round(np.sum((y_true==y_pred))/len(y_true),2)
