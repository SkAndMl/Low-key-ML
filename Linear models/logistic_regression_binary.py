import numpy as np

class LogisticRegressionBC:

    """
    Binary classifier logistic regression model has been implemented here.
    For complete explanation take a look at the article link that is available in the README of repo.
    """

    def __init__(self, lr=0.01,random_state=42,epochs=50):

        """
        :param lr: learning rate.
        :param random_state: seed for random initialization of weights.
        :param epochs: number of times the model sees the training data.
        """

        self.lr = lr
        self.random_state = random_state
        self.epochs = epochs

    def fit(self,X,y):

        rgen = np.random.RandomState(self.random_state)
        self.weights = rgen.normal(loc=0.0,size=X.shape[1],scale=0.0)
        self.cost = []

        for _ in range(self.epochs):
            np.seterr(divide='ignore')
            input = self.net_input(X)
            output = self.activation(X)
            errors = y-output
            self.weights[1:] += self.lr*X.T.dot(errors)
            self.w[0] += self.lr*errors.sum()
            c = (-y.dot(np.log(output)))-((1-y).dot(np.log(1-output)))
            self.cost.append(c)
        return self

    def predict(self,X):
        inp = self.net_input(X)
        output = self.activation(inp)
        return np.wher(output>=9.5,1,0)

    def net_input(self,X):
        return np.dot(X,self.weights[1:])+self.weights[0]

    def activation(self,X):
        return 1/(1+np.exp(-np.clip(X,-250,250)))