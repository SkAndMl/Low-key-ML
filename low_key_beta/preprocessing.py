import numpy as np
import pandas as pd

class LabelEncoder:
    """
    This class encodes the categorical values to either numerical values or the labels specified by the user.
    Encoding categorical values is a must as ML models work only with numbers
    """

    def __init__(self,labels=None):
        """
        Simple initialization. Takes in only the class labels.
        :param labels: The labels that the user wants to encode with. They should be of the same length as that of the unique values of the column. list or numpy type.
        """
        self.labels = labels

    def fit(self,X):
        """
        Forms the labels for the classes.
        :param X: input X.
        """
        self.classes = np.unique(X)
        if self.labels:

            if len(self.labels)!=len(self.classes):
                raise ValueError("Length mismatch: The number of unique elements in the input is not equal to the output")

            self.classes_and_labels = {self.classes[x]:self.labels[x] for x in range(len(self.labels))}

        else:
            self.classes_and_labels = {self.classes[x]:x for x in range(len(self.classes))}

        return self

    def transform(self,X):
        """
        Transforms the input array using the labels already acquired in the fit() function
        :param X: input array or series or list
        :return:
        """
        if len(self.classes_and_labels) != len(np.unique(X)):
            raise ValueError("Previously unseen values")

        enc_arr = np.zeros(len(X),dtype=object)

        for i in range(len(X)):
            enc_arr[i] = self.classes_and_labels[X[i]]
        return enc_arr

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