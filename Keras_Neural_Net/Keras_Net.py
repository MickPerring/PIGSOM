'''
* Full Keras_Net.py program

* Author: Mick Perring
* Date: October 2018

* Adapted from code by Jason Brownlee in his online tutorial:
* Develop Your First Neural Network in Python With Keras Step-By-Step
* J. Brownlee, "Develop Your First Neural Network in Python With Keras Step-By-Step",
* Machine Learning Mastery, 2016.
* https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

* This program creates a neural network using the Keras machine learning
* libraries for Python, run on the TendorFlow deep learning framework. The
* program is developed to work off of the Pima Indians Diabetes database,
* which is hard coded as the input filename
'''

from keras.models import Sequential
from keras.layers import Dense
import numpy

'''
* This function simply loads the data file (.csv format) into the program,
* and partitions it into input and output data to be used by the program.
* The output data is the last column of the .csv files, containing the
* expected outputs of each line of training data.
'''
def data_load(file):
	dataset = numpy.loadtxt(file, delimiter=";")
	x = dataset[:,0:-1]
	y = dataset[:,-1]
	return x, y

'''
* The main function of the program. Builds the neural network using the input
* parameters set by the user. Keras has built in methods to add network layers
* one by one in single lines of code, each with their own number of neurons
* and choice of activation function. There are also built in Keras methods to
* analyse and output the performance of the network to the console.
'''
def neural_net(in_layer, h_layers, out_layer, num_layers, epoch, batch, x, y):

	model = Sequential()

	# add first hidden layer
	model.add(Dense(h_layers[0][0], input_dim=in_layer, activation=h_layers[0][1]))

	# add any additional hidden layers
	if(num_layers > 2):	
		for i in range(1, num_layers-1):
			model.add(Dense(h_layers[i][0], activation=h_layers[i][1]))

	# add output layer
	model.add(Dense(out_layer[0], activation=out_layer[1]))

	# compile and run model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(x, y, epochs=epoch, batch_size=batch)

	# output performance results
	scores = model.evaluate(x, y)
	print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# called when the program is run
if __name__ == '__main__':

	# load the dataset
	X, Y = data_load("pima-indians-diabetes.csv")

	# input parameters - change manually if desired
	input_layer = 8
	hidden_layers = [[400, 'sigmoid'], [400, 'sigmoid'], [400, 'sigmoid']]
	num_layers = len(hidden_layers)
	output_layer = [1, 'sigmoid']
	epochs = 200
	batch_size = 10

	# create and run neural network
	neural_net(input_layer, hidden_layers, output_layer, num_layers, epochs, batch_size, X, Y)