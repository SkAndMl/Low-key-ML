import numpy as np

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
