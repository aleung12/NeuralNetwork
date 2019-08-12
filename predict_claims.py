from activation_functions import *
from load_data import load_data
import pandas as pd
import numpy as np



class TrainedNeuralNetwork:

    def __init__(self, x):
        self.input = x

    def load_weights(self, weights=None):
        if weights is None:
            self.weights0 = np.load('state/best/weights0.npy')
            self.weights1 = np.load('state/best/weights1.npy')
            self.weights2 = np.load('state/best/weights2.npy')
            self.biasw1   = np.load('state/best/biasw1.npy')
            self.biasw2   = np.load('state/best/biasw2.npy')
        else:
            self.weights0, self.weights1, self.weights2, self.biasw1, self.biasw2 = weights

    def feed_forward(self):
        self.layer1 = leaky_ReLU(self.input.dot(self.weights0))
        self.layer2 = leaky_ReLU(self.layer1.dot(self.weights1) + np.ones((self.input.shape[0],1)).dot(self.biasw1))
        self.output = sigmoid(self.layer2.dot(self.weights2) + np.ones((self.input.shape[0],1)).dot(self.biasw2))



def predict_claims(nn_input, flight_id):

    nn = TrainedNeuralNetwork(nn_input)

    print('Importing weights...')
    try: nn.load_weights()
    except IOError: print('Weights cannot be found in ./state/')

    print('Calculating claim amounts...')
    nn.feed_forward()
    claim_amt = 800 * nn.output

    print('Saving results...')
    filename = 'predicted_claim_amounts.csv'
    df = pd.DataFrame({'flight_id':flight_id, 'predicted_claim_amt':claim_amt.round(3).flatten()})
    df.to_csv(filename, index=False)

    print('Finished. Results saved to '+filename)
    


if __name__ == '__main__':

    try: 
        try: 
            filename = input('Enter path to flight delays data (hit Return for \'./flight_delays_data.csv\'): ')
        except SyntaxError: 
            filename = 'flight_delays_data.csv'
    
        print('Loading data...')
        nn_input, flight_id = load_data(filename, training=False)
  
    except IOError: 
        print('File %s does not exist' % filename)

    predict_claims(nn_input, flight_id)


