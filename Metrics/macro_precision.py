from metrics import true_positive,false_positive
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


