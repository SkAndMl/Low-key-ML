import numpy as np
from sklearn.base import BaseEstimator


class LinearRegressionSGD(BaseEstimator):
    """
      Linear regression using SGD optimization algorithm.
      SGD improves on the execution time of GD by introducing randomness in the training process.
      Please find the full explanation for this algorithm in link provided in the README section of the repo.
      You can find some useful implementation of this model in GD.ipynb
    """

    def __init__(self, lr=0.01, epochs=100, random_state=42):

        """
        :param lr: learning rate.
        :param epochs: number of times the model can see the training data.
        :param random_state: seed for random initialization of weights.
        """

        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X, y):
        np.random.seed(self.random_state)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # adding the bias column
        self.theta = np.random.randn(X_b.shape[1], 1)  # random initialization.
        for _ in range(self.epochs):
            for i in range(len(X_b)):
                random_ind = np.random.randint(len(X_b))  # random index.
                X_random = X_b[random_ind:random_ind + 1]  # random training index.
                y_random = y[random_ind:random_ind + 1]
                output = X_random.dot(self.theta)  # hypothesis or net input.
                gradient = 2 * X_random.T.dot(output - y_random)  # the gradient
                self.theta = self.theta - self.lr * gradient  # descent step.
        self.intercept_ = self.theta[0][0]
        self.coef_ = np.ravel(self.theta[1:])
        return self

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)