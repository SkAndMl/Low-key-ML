from metrics import true_positive,false_positive

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