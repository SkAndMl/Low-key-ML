import numpy as np

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

