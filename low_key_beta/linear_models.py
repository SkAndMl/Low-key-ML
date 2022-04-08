import numpy as np
from sklearn.base import BaseEstimator

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

    def __str__(self):
        return "LogisticRegressionBC"

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

class LinearRegressionMBGD(BaseEstimator):
    """
    Linear regression using Mini-Batch GD optimization algorithm,
    MBGD uses the best characteristics of gradient descent and stochastic gradient descent.
    MBGD learns in batches which speeds up the program when the dataset is really large.
    """
    def __init__(self,lr=0.01,epochs=100,random_state=42,batch_size=32):

        """
        :param lr: learning rate, specifies the magnitude of GD's step
        :param epochs: the number of times you want the model to see the training data
        :param random_state: seed for initialization of weights. Used for reproduciblility.
        :param batch_size: the size of the batch that the model will train on a particular instance.
        """
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state
        self.batch_size = batch_size

    def fit(self,X,y):
        np.random.seed(self.random_state)
        X_b = np.c_[np.ones((X.shape[0],1)),X]
        self.theta = np.random.randn(X_b.shape[1],1)
        for _ in range(self.epochs):
            for i in range(len(X_b)):
                random_ind = np.random.randint(len(X_b)-self.batch_size)
                X_random = X_b[random_ind:random_ind+self.batch_size]
                y_random = y[random_ind:random_ind+self.batch_size]
                output = X_random.dot(self.theta)
                gradient = (2/self.batch_size)*X_random.T.dot(output-y_random)
                self.theta = self.theta - self.lr*gradient
        self.intercept_ = self.theta[0][0]
        self.coef_ = np.ravel(self.theta[1:])
        return self

    def predict(self,X):
        X_b = np.c_[np.ones((X.shape[0],1)),X]
        return X_b.dot(self.theta)