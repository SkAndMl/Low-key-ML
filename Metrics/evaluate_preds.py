import numpy as np
def evaluate_preds(y_true, y_pred):
    """
    Function to evaluate time series predictions
    Calculates mae, mse, rmse, mape and mase and returns 
    dictionary format
    """
    eval_dict = {}
    eval_dict['mae'] = np.mean(np.absolute(y_true-y_pred))
    eval_dict['mse'] = np.mean(np.absolute(y_true-y_pred)**2)
    eval_dict['rmse'] = np.sqrt(eval_dict['mse'])
    eval_dict['mase'] = eval_dict['mae']/np.mean(np.absolute(y_true[1:]-y_true[:-1]))
    eval_dict['mape'] = np.mean(100*np.absolute(y_true-y_pred)/y_true)
    return eval_dict

if __name__ == '__main__':
    y = np.linspace(0,1,97)
    eval_dict = evaluate_preds(y[1:], y[:-1])
    for key in eval_dict.keys():
        print(f"{key}: {eval_dict[key]}")

