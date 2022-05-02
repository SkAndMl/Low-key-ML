def r2(y_true,y_pred):
    import numpy as np
    mean_y_true = np.mean(y_true)
    num = np.sum(np.square(y_true-y_pred))
    den = np.sum(np.square(y_true-mean_y_true))
    return 1 - (num/den)