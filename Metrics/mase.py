import numpy as np
def mean_absolute_scaled_error(y_true, y_pred):
    """
    Implemenation of mean absolute scaled error.
    MASE is a metric used for time series predictions.
    """
    numerator = np.mean(np.absolute(y_true-y_pred))
    denominator = np.mean(np.absolute(y_true[1:]-y_true[:-1]))
    return numerator/denominator

if __name__ == '__main__':
    y = np.linspace(0,1,101)
    mase = mean_absolute_scaled_error(y[1:], y[:-1])
    print(f"Sample MASE: {mase}")
