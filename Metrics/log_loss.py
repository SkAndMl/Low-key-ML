def log_loss(y_true,y_proba):
    import numpy as np
    avg_loss = 0
    for i,j in zip(y_true,y_proba):
        log_loss = -1*(i*np.log(j) + (1-i)*np.log(1-j))
        avg_loss += log_loss

    return avg_loss/len(y_true)