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

        X_b = np.c_[np.ones((len(X),1)),X]
        rgen = np.random.RandomState(self.random_state)
        self.weights = rgen.normal(loc=0.0,size=X_b.shape[1],scale=0.0)
        self.cost = []

        for _ in range(self.epochs):
            np.seterr(divide='ignore')
            input = self.net_input(X_b)
            output = self.activation(X_b)
            errors = y-output
            self.weights += self.lr*X_b.T.dot(errors)
            c = (-y.dot(np.log(output)))-((1-y).dot(np.log(1-output)))
            self.cost.append(c)
        return self

    def predict(self,X):
        X_b = np.c_[np.ones((len(X),1)),X]
        inp = self.net_input(X_b)
        output = self.activation(inp)
        return np.where(output>=0.5,1,0)

    def net_input(self,X):
        return X.dot(self.weights[:,np.newaxis])

    def activation(self,X):
        X_arr = 1/(1+np.exp(-np.clip(X,-250,250)))
        return np.mean(X_arr,axis=1)
