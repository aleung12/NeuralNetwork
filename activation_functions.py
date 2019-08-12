import numpy as np


## leaky rectified linear unit activiation function
leaky_ReLU = lambda x : np.max([0.01*x, x], axis=0)

def leaky_ReLU_derivative(x):
    out = np.ones(x.shape)
    out[np.where(x<0)] = 0.01
    return out


## sigmoid activation function
sigmoid = lambda x : (1 + np.exp(-x))**-1

def sigmoid_derivative(x):
    sigmoid_x = sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)

