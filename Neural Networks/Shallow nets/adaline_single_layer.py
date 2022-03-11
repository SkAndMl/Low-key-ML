import numpy as np

class Adaline:

    """
    Adaline neural net(binary classifier) with a single layer has been implemented in this script.
    Please find the explanation for this script in link provided in the README of the repo.
    """

    def __init__(self, lr=0.01, random_state=1, n_iter=50):
        self.lr = lr
        self.random_state = random_state
        self.n_iter = n_iter


    def net_input(self, X):
        return np.dot(X, self.w[1:]) + self.w[0]


    def activation(self, X):
        return X


    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc=0.0, size=X.shape[1] + 1, scale=0.01)
        self.cost = []

        for _ in range(self.n_iter):
            inp = self.net_input(X)
            output = self.activation(inp)
            errors = y - output
            self.w[1:] += self.lr * X.T.dot(errors)
            self.w[0] += self.lr * errors.sum()
            c = (errors ** 2).sum() / 2
            self.cost.append(c)
        return self


    def predict(self, X):
        inp = self.net_input(X)
        output = self.activation(inp)
        return np.where(output > 0.0, 1, -1)