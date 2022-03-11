import numpy as np
import pandas as pd

def train_test_split(X,y,test_size=0.2,random_state=42):

    """
    Accepts only a dataframe or a numpy array as input.
    :param X: input data X
    :param y: input data y
    :param test_size: specifies the size of the test dataset.
    :param random_state: seed for shuffling the data
    :return: X_train,X_test,y_train,y_test
    """

    np.random.seed(random_state)
    shuffled_index = np.random.permutation(len(X))
    train_indices = shuffled_index[:int(len(X)*(1-test_size))]
    test_indices = shuffled_index[int(len(X)*test_size):]
    if type(X)==type(pd.DataFrame()):
        X_train,X_test,y_train,y_test = X.iloc[train_indices],X.iloc[test_indices],y.iloc[train_indices],y.iloc[test_indices]
        return X_train, X_test, y_train, y_test
    elif type(X)==type(np.array()):
        X_train,X_test,y_train,y_test = X[train_indices],X[test_indices],y[train_indices],y[test_indices]
        return X_train, X_test, y_train, y_test
    else:
        raise TypeError("Only dataframes and numpy arrays are accepted as input")

