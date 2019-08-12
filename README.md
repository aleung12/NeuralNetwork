# NeuralNetwork
Predicting flight cancellations with a generic two-layer Artificial Neural Network


[```neural_network.py```](neural_network.py) and 
[```activation_functions.py```](activation_functions.py) 
are generic. The artificial neural network is set to initialize to two layers, with 
400 neurons in the first hidden layer and 250 neurons in the second hidden layer.
This neural network architecture is configured to work with 
[flight delays data](https://drive.google.com/a/terminal1.co/file/d/1AkEc76q6NbqEojk3BQJEfbx-RIigDCve/), 
which [```load_data.py```](load_data.py) transforms into 385 binary input variables. 

The neural network uses a 
[leaky rectified linear unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Leaky_ReLUs) 
activation function to compute the hidden layers, and a sigmoid activation function to 
compute the output layer (which is a binary classification in the case of predicting 
flight cancellations). It is straightforward to modify the specified activation function 
for the output layer (don't forget the associated derivative in back propagation), though 
care needs to be taken as the learning rate likely requires a lower initial value if 
the network is adapted to perform regression.


- To train the artificial neural network using _k_-fold cross-validation, execute 
  [```train_network.py```](train_network.py) in command line 
  (_k_ = 2 is a single train/test split; _k_ >= 2):
  ```
  python train_network.py 2
  ```
  The training data file './flight_delays_data.csv' is required. The user can specify 
  a different file at line 112. The program defaults to starting a new training from 
  scratch. To resume a training, modify line 115 accordingly and specify k at line 116 
  for the k-fold cross-validation in progress.


- To use a trained neural network to predict claims for delayed or cancelled flights, 
  execute [```predict_claims.py```](predict_claims.py) in command line:
  ```
  python predict_claims.py
  ```
  The user will be prompted to specify the path to a .csv file with the test data. 
  Weights for the trained network needs to be in [./state/best/](./state/best).
