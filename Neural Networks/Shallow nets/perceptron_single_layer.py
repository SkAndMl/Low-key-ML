import numpy as np


class Perceptron:

    """
    Perceptron single layer neural network is implemented in this script.
    Please find the explanation for the code in the link provided in the README of the repo
    """

    def __init__(self, eta=0.01, random_state=1, epochs=50):
        """
        :param eta: learning rate
        :param random_state: seed for random initialization of weights.
        :param epochs: number of times the model goes through the data.
        """
        self.eta = eta
        self.random_state = random_state
        self.epochs = epochs

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.epochs):
            errors = 0
            for a, b in zip(X, y):
                update = self.eta * (b - self.predict(a))
                self.w_[1:] += update * a
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(self.w_[1:].T, X) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, -1)