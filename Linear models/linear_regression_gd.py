import numpy as np
from sklearn.base import BaseEstimator


class LinearRegressionGD(BaseEstimator):
    """
      Linear regression using Gradient Descent optimization algorithm.
      The important part in this script is three lines in the 'for loop' which implement Gradient Descent.
    """

    def __init__(self, lr=0.01, epochs=100, random_state=42):

        """
        :param lr: learning rate, specifies the magnitude of GD's step
        :param epochs: the number of times you want the model to see the training data
        :param random_state: seed for initialization of weights. Used for reproduciblility.
        """

        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X, y):
        np.random.seed(self.random_state)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.theta = np.random.randn(X_b.shape[1], 1)  # random weight initialization
        for _ in range(self.epochs):
            output = X_b.dot(self.theta)  # the hypothesis or net input
            gradient = (2 / X_b.shape[0]) * X_b.T.dot(output - y)  # calculating the gradient
            self.theta = self.theta - self.lr * gradient
        self.intercept_ = self.theta[0][0]
        self.coef_ = np.ravel(self.theta[1:])  # these are the params that you can look at after training/
        return self

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)