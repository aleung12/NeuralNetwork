# NeuralNetwork
Predicting flight cancellations with a generic two-layer Artificial Neural Network


[```neural_network.py```](neural_network.py) and 
[```activation_functions.py```](activation_functions.py) 
are generic. The artificial neural network is set to initialize to two layers, with 
400 neurons in the first hidden layer and 250 neurons in the second hidden layer.
This neural network architecture is designed to work with the flight delays data, which 
[```load_data.py```](```load_data.py```) transforms into 385 binary input variables. 
The neural network uses a 
[leaky rectified linear unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Leaky_ReLUs) 
activation function to compute the hidden layers, and a sigmoid activation function to 
compute the output layer (which is a binary classification in the case of predicting 
flight cancellations). It is straightforward to modify the specified activation function 
for the output layer (don't forget the associated derivative in back propagation), though 
care needs to be taken as the learning rate likely requires a lower initial value if 
the network is adapted to perform regression.


- To train the artificial neural network, execute 
  [```train_network.py```](train_network.py) in command line:
  ```
  python train_network.py
  ```
  The training data file './flight_delays_data.csv' is required (the file name is 
  specified at line 111). The file defaults to resuming training from the state provided 
  in [./state/](./state) (to start training from scratch, modify line 114).


- To use a trained neural network to predict claims for delayed or cancelled flights, 
  execute [```predict_claims.py```](predict_claims.py) in command line:
  ```
  python predict_claims.py
  ```
  The user will be prompted to specify the path to a .csv file with the test data. 
  Weights for the trained network needs to be in [./state/best/](./state/best).
