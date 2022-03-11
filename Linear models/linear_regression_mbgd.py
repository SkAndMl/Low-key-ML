import numpy as np
from sklearn.base import BaseEstimator
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