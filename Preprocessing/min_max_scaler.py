import numpy as np
import pandas as pd

class MinMaxScaler:
    """
    MinMaxScaler is used to normalize the input array or dataframe such that the array values get into the range-[0,1].
    Note: This version of MinMaxScaler only accepts a dataframe or an array of as input.
    :return: It returns a multidimensional array
    """
    def fit(self,X):

        if type(X)!=type(pd.DataFrame()) and type(X)!=type(np.array([1,2])): # checks for the datatype
            raise TypeError(f"MinMaxScaler accepts either a dataframe or a numpy array as input. It does not accept {type(X)} as input dtype")

        if type(X)==type(pd.DataFrame()):
            X = X.values #  gets the numpy array from the DataFrame
            self.min_X,self.max_X = np.zeros(X.shape[1]),np.zeros(X.shape[1])
            for i in range(X.shape[1]):
                self.min_X[i],self.max_X[i] = np.min(np.squeeze(X[:,i])),np.max(np.squeeze(X[:,i]))
        else:
            req_arr = np.squeeze(X)
            self.min_X,self.max_X = np.min(req_arr,axis=0),np.max(req_arr,axis=0)

        return self

    def transform(self,X):
        """
        :param X: input array or dataframe
        :return: returns a normalized multidimensional numpy array.
        Important note: MinMaxScaler assumes that the input dataframe has its columns in the same order as the one passed to the fit method .
        """
        if type(X)==type(pd.DataFrame()):
            if X.shape[1]!=len(self.min_X):
                raise ValueError("Length mismatch: The transformer was trained on a different length") # checks for the number of features and if they don't match it outputs an error

        if type(X)==type(pd.DataFrame()):

            new_X = np.zeros(X.shape)
            X = X.values

            for i in range(X.shape[1]):
                new_X[:,i] = (np.squeeze(X[:,i])-self.min_X[i])/(self.max_X[i]-self.min_X[i])

        else:
            X = np.squeeze(X)
            new_X = (X-self.min_X)/(self.max_X-self.min_X)

        return new_X