from metrics import true_positive,false_positive
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