from activation_functions import *
import numpy as np



class NeuralNetwork:

    def __init__(self, x, y):
        self.input = x
        self.y     = y
        
        self.n1 = 400                   ## number of neurons in first hidden layer
        self.n2 = 250                   ## number of neurons in second hidden layer
        self.learning_rate = 3e-5       ## initial learning rate (back propagation step size)
        
        self.weights0 = np.random.normal(0, 0.1, (self.input.shape[1], self.n1))
        self.weights1 = np.random.normal(0, 0.1, (self.n1, self.n2))
        self.weights2 = np.random.normal(0, 0.1, (self.n2, 1))
        self.biasw1   = np.random.normal(0, 0.1, (1, self.n2))
        self.biasw2   = np.random.normal(0, 0.1, (1, 1))

        
    def feed_forward(self):
        self.layer1 = leaky_ReLU(self.input.dot(self.weights0))
        self.layer2 = leaky_ReLU(self.layer1.dot(self.weights1) + np.ones((self.input.shape[0],1)).dot(self.biasw1))
        self.output = sigmoid(self.layer2.dot(self.weights2) + np.ones((self.input.shape[0],1)).dot(self.biasw2))


    def back_prop(self, testing=True):
        # apply chain rule to find derivative of the loss function with respect to weights
        chain_rule2 = 2 * (self.y - self.output) * sigmoid_derivative(self.output)
        d_weights2  = self.layer2.T.dot( chain_rule2 )
        d_biasw2    = np.ones((1,self.input.shape[0])).dot( chain_rule2 )
        
        chain_rule1 = chain_rule2.dot( self.weights2.T ) * leaky_ReLU_derivative(self.layer2)
        d_weights1  = self.layer1.T.dot( chain_rule1 )
        d_biasw1    = np.ones((1,self.input.shape[0])).dot( chain_rule1 )

        chain_rule0 = chain_rule1.dot( self.weights1.T ) * leaky_ReLU_derivative(self.layer1)
        d_weights0  = self.input.T.dot( chain_rule0 )
        
        # update the weights with derivative of the loss function modulo the learning rate
        self.weights0 += self.learning_rate * d_weights0
        self.weights1 += self.learning_rate * d_weights1
        self.weights2 += self.learning_rate * d_weights2

        self.biasw1   += self.learning_rate * d_biasw1
        self.biasw2   += self.learning_rate * d_biasw2

        if testing:
            print('\n                               first                   third              ')
            print('                    min       quartile     median     quartile      max     ')
            print('                 ----------  ----------  ----------  ----------  ---------- ')
            print('    d_weights0 |%11s %11s %11s %11s %11s' % quartiles(d_weights0))
            print('    d_weights1 |%11s %11s %11s %11s %11s' % quartiles(d_weights1))
            print('    d_weights2 |%11s %11s %11s %11s %11s' % quartiles(d_weights2))

            print('    d_biasw1   |%11s %11s %11s %11s %11s' % quartiles(d_biasw1))
            print('    d_biasw2   |%11s %11s %11s %11s %11s' % quartiles(d_biasw2))


    def loss_function(self):
        return np.mean(np.square(self.y - self.output))


    def save_weights(self, best=False):
        if best: path = 'state/best/'
        else:    path = 'state/'
        np.save(path+'weights0.npy', self.weights0)
        np.save(path+'weights1.npy', self.weights1)
        np.save(path+'weights2.npy', self.weights2)
        np.save(path+'biasw1.npy',   self.biasw1)
        np.save(path+'biasw2.npy',   self.biasw2)



## quartiles for testing
quartiles = lambda x : ('%.3e'%np.min(x), '%.3e'%np.percentile(x,25), '%.3e'%np.median(x), '%.3e'%np.percentile(x,75), '%.3e'%np.max(x))

