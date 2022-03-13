import numpy as np
import pandas as pd


class StandardScaler:
    """
    StandardScaler is used to scale the input array or dataframe such that the array values have a mean of 0 and a standard deviation of 1.
    Note: This version of StandardScaler only accepts a dataframe or an array of as input.
    :return: It returns a multidimensional array
    """
    def fit(self,X):

        if type(X)!=type(pd.DataFrame()) and type(X)!=type(np.array([1,2])): # checks for the datatype
            raise TypeError(f"StandardScaler accepts either a dataframe or a numpy array as input. It does not accept {type(X)} as input dtype")

        if type(X)==type(pd.DataFrame()):
            X = X.values #  gets the numpy array from the DataFrame
            self.mean_X,self.std_X = np.zeros(X.shape[1]),np.zeros(X.shape[1])
            for i in range(X.shape[1]):
                req_arr = np.squeeze(X[:,i])
                self.mean_X[i],self.std_X[i] = np.mean(req_arr,axis=0),np.std(req_arr,axis=0) #  computes the mean and std of each feature.
        else:
            req_arr = np.squeeze(X)
            self.mean_X,self.std_X = np.mean(req_arr,axis=0),np.std(req_arr,axis=0)

        return self

    def transform(self,X):
        """
        :param X: input array or dataframe
        :return: returns a scaled multidimensional numpy array.
        Important note: StandardScaler assumes that the input dataframe has its columns in the same order as the one passed to the fit method .
        """
        if type(X)==type(pd.DataFrame()):
            if X.shape[1]!=len(self.mean_X):
                raise ValueError("Length mismatch: The transformer was trained on a different length") # checks for the number of features and if they don't match it outputs an error

        if type(X)==type(pd.DataFrame()):

            new_X = np.zeros(X.shape)
            X = X.values

            for i in range(X.shape[1]):
                new_X[:,i] = (np.squeeze(X[:,i])-self.mean_X[i])/self.std_X[i]

        else:
            X = np.squeeze(X)
            new_X = (X-self.mean_X)/self.std_X

        return new_X

