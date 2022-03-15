import numpy as np

def f1_score(y_true,y_pred):
    """
    Calculates the f1_score which is one of the metrics used to evaluate a classification model
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: returns the f1_score of the classification model using the formula f1_score = (2*precision*recall)/(precision+recall)
    """
    if(len(y_true)!=len(y_pred)):
        raise ValueError("Length mismatch: Lengths of y_true and y_pred don't match")

    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    total_positve = np.sum(y_pred)
    true_positive = np.sum((y_pred==1)&(y_true==y_pred))
    precision = np.round(true_positive/total_positve,2)

    total_positve = np.sum(y_true)
    recall = np.round(true_positive/total_positve,2)
    f1_score = (2*(precision*recall))/(precision+recall)
    return np.round(f1_score,2)


