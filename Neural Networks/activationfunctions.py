import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def tanh(x):
    return 2*sigmoid(2*x)-1

def leakyrelu(x,alpha=0.01):
    return np.maximum(alpha*x,x)

def elu(x,alpha=1):
    return np.where(x<0,alpha*(np.exp(x)-1),x)