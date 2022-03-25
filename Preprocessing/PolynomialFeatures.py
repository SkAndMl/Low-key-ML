import numpy as np
import pandas as pd
from itertools import combinations

class PolynomialFeatures:

    def __init__(self,degree=2,interaction_only=False):

        self.degree = degree
        self.interaction_only = interaction_only

    def fit_transform(self,X):

        if type(X)!=type(np.array([1,2])):
            raise TypeError("PolynomialFeatures accepts only numpy array as input")

        try:
            shape_X = X.shape[1]
        except IndexError:
            X = X.reshape(len(X),1)

        poly_X = np.ones(len(X))
        for i in range(X.shape[1]):
            poly_X = np.c_[poly_X, X[:, i]]
            if not self.interaction_only:
                poly_X = np.c_[poly_X,np.power(X[:,i],self.degree)]

        for r in range(2,self.degree+1):
            if r<=X.shape[1]:
                print(r)
                combs = combinations(np.arange(0,X.shape[1]),r)
                for comb in combs:
                    req_X = np.ones((len(X),1))
                    for c in comb:
                        req_X = np.squeeze(req_X)*np.squeeze(X[:,c])
                        req_X = req_X.reshape(len(X),1)
                    poly_X = np.c_[poly_X,req_X]
            else:
                print("Warning: Given degree is more than the columns in the input value")

        return poly_X
