import Keras_Net as kn
from keras.models import Sequential
from keras.layers import Dense
import numpy

# load the dataset
X, Y = kn.data_load("pima-indians-diabetes.csv")

# input parameters - change manually if desired
input_layer = 8
hidden_layers = [[400, 'sigmoid'], [400, 'sigmoid'], [400, 'sigmoid']]
num_layers = len(hidden_layers)
output_layer = [1, 'sigmoid']
epochs = 200
batch_size = 10

# create and run neural network
kn.neural_net(input_layer, hidden_layers, output_layer, num_layers, epochs, batch_size, X, Y)