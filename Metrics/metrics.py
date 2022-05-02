import numpy as np
import pandas as pd

def mae(y_true,y_pred):
    """
    MAE is an evaluation metric that is used to evaluate a regression model
    :param y_true: true continuous values.
    :param y_pred: predicted values
    :return: returns MAE calculated using (y_true-y_pred)/len(y_true)
    """
    if len(y_true)!=len(y_pred):
        raise ValueError(f"Length mismatch: The lengths of y_true({len(y_true)}) and y_pred({len(y_pred)}) do not match")

    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    error = np.sum(np.abs(y_pred-y_true))  # total error.

    return np.round(error/len(y_true),3)

def mse(y_true,y_pred):

    if (len(y_true)!=len(y_pred)):
        raise ValueError(f"Length mismatch: The length of y_true ({len(y_true)}) is not the same as that of y_pred ({len(y_pred)})")

    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    error = np.square(y_true-y_pred)

    if (np.array(len(error[0]))>1):
        error = np.mean(error,axis=1)

    error = np.sum(error)

    return np.round(error/len(y_true),3)

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

import numpy as np

def confusion_matrix_binary(y_true,y_pred):
    """
    Confusion matrix is an evaluation metric that is used to output the values of true positive and false negative (row wise).
    In this function the actual classes come along the rows.
    :param y_true: true class labels (numpy array)
    :param y_pred: predicted class labels (numpy array)
    :return: Returns the unique classes which points out the row-order of confusion matrix and the confusion matrix
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true)!=len(y_pred):
        raise ValueError(f"Length mismatch: Length of y_true ({len(y_true)}) and length of y_pred ({len(y_pred)}) are different")

    if len(np.unique(y_true))>2 or len(np.unique(y_pred))>2:
        raise ValueError("Unique classes count error: Only 2 unique classes must be present")

    unq_classes = np.unique(y_true)
    conf_mat = np.zeros((2,2))

    for i in range(len(unq_classes)):
        true_pos_sum = 0
        false_neg_sum = 0

        true_pos_sum = np.sum((y_true==y_pred)&(y_true==unq_classes[i]))
        false_neg_sum = np.sum(y_true==unq_classes[i]) - true_pos_sum

        conf_mat[i,i] = true_pos_sum
        if i==0:
            conf_mat[i,1] = false_neg_sum
        else:
            conf_mat[i,0] = false_neg_sum

    return unq_classes,conf_mat

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
    true_positive = np.sum((y_pred==1)&(y_pred==y_true))

    return np.round(true_positive/total_positive,2)


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
    true_positive = np.sum((y_pred==1)&(y_pred==y_true))

    return np.round(true_positive/positive,2)

def classification_report(y_true,y_pred):
    """
    Classification report lists precision, recall and f1_score of the passed y_true and y_pred labels
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: returns a dataframe consisting of recall,precision and f1_score as columns and the class labels as index
    """
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    if len(y_pred)!=len(y_true):
        raise ValueError("Length mismatch: The length of y_true is not same as the length of y_pred")

    # creating the dataframe
    classes = np.unique(y_true)  # gets the unique classes
    columns = ['precision','recall','f1_score']  # required columns
    data = np.zeros((len(classes),3))  # data for the dataframe
    report_df = pd.DataFrame(data=data,index=classes,columns=columns)  # classification_report dataframe

    for i in range(len(classes)):
        y_true_mod = np.where(y_true!=classes[i],0,1)  # using OneVsRest(OVR) principle
        y_pred_mod = np.where(y_pred!=classes[i],0,1)  # using OneVsRest(OVR) principle
        report_df.iat[i,0] = precision_binary(y_true=y_true_mod,y_pred=y_pred_mod)  # precision for the i th class label
        report_df.iat[i,1] = recall_binary(y_true=y_true_mod,y_pred=y_pred_mod)  # recall for the ith class label
        report_df.iat[i,2] = f1_score(y_true=y_true_mod,y_pred=y_pred_mod)  # f1_score for the ith class label

    return report_df

def true_positive(y_true,y_pred):

    import numpy as np
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sum((y_true==y_pred)&(y_true==1))

def false_positive(y_true,y_pred):

     import numpy as np
     y_true = np.array(y_true)
     y_pred = np.array(y_pred)
     return np.sum((y_pred==1)&(y_true==0))

def false_negative(y_true,y_pred):

    import numpy as np
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sum((y_pred==0)&(y_true==1))

def micro_precision(y_true,y_pred):
    import numpy as np
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    unq_classes = np.unique(y_true)
    tp,fp = 0,0
    for c in unq_classes:
        tp += true_positive(y_true==c,y_pred==c)
        fp += false_positive(y_true==c,y_pred==c)
    return tp/(tp+fp)

def macro_precision(y_true,y_pred):
    import numpy as np
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    unq_classes = np.unique(y_true)
    prec_classes = []
    for c in unq_classes:
        tp = true_positive(y_true==c,y_pred==c)
        fp = false_positive(y_true==c,y_pred==c)
        precision_c = tp/(tp+fp)
        prec_classes.append(precision_c)
    return np.mean(prec_classes)

def weighted_precision(y_true,y_pred):
    import numpy as np
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    unq_classes = np.unique(y_true)
    len_classes = [np.sum(y_true==c) for c in unq_classes]
    prec_classes = []
    for c in unq_classes:
        tp = true_positive(y_true == c, y_pred == c)
        fp = false_positive(y_true == c, y_pred == c)
        precision_c = tp / (tp + fp)
        prec_classes.append(precision_c)
    prec_classes = np.array(prec_classes)
    return np.average(prec_classes,weights=len_classes)

def r2(y_true,y_pred):
    import numpy as np
    mean_y_true = np.mean(y_true)
    num = np.sum(np.square(y_true-y_pred))
    den = np.sum(np.square(y_true-mean_y_true))
    return 1 - (num/den)