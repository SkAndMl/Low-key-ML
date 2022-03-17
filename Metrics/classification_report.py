import numpy as np
import pandas as pd
from precision_binary import precision_binary
from recall_binary import recall_binary
from f1_score import f1_score
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