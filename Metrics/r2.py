def r2(y_true,y_pred):
    """
    y_true -> true labels
    y_pred -> predicted labels
    R2 defines how well does the model fit the data.
    """
    import numpy as np
    mean_y_true = np.mean(y_true)
    num = np.sum(np.square(y_true-y_pred))
    den = np.sum(np.square(y_true-mean_y_true))
    return 1 - (num/den)