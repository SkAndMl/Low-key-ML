import numpy as np

def confusion_matrix_binary(y_true,y_pred):
    """
    Confusion matrix is an evaluation metric that is used to output the values of true positive and false negative (row wise).
    In this function the actual classes come along the rows.
    :param y_true: true class labels
    :param y_pred: predicted class labels
    :return: Returns the unique classes which points out the row-order of confusion matrix and the confusion matrix
    """
    if len(y_true)!=len(y_pred):
        raise ValueError(f"Length mismatch: Length of y_true ({len(y_true)}) and length of y_pred ({len(y_pred)}) are different")

    if len(np.unique(y_true))>2 or len(np.unique(y_pred))>2:
        raise ValueError("Unique classes count error: Only 2 unique classes must be present")

    unq_classes = np.unique(y_true)
    conf_mat = np.zeros((2,2))

    for i in range(len(unq_classes)):
        true_pos_sum = 0
        false_neg_sum = 0
        for j in range(len(y_pred)):
            if  y_pred[j]==unq_classes[i] and y_true[j]==unq_classes[i]:
                true_pos_sum+=1
            elif y_pred[j]!=unq_classes[i] and y_true[j]==unq_classes[i]:
                false_neg_sum+=1
            else:
                continue
            conf_mat[i,i] = true_pos_sum
            if i==0:
                conf_mat[i,1] = false_neg_sum
            else:
                conf_mat[i,0] = false_neg_sum

    return unq_classes,conf_mat

