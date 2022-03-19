import numpy as np

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

