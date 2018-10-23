'''
* Full NeuralNetwork.py program

* Author: Mick Perring
* Date: October 2018

* Adapted from code by Michael A. Nielsen in his online book:
* Neural Networks and Deep Learning
* M. Nielsen, Neural Networks and Deep Learning. Determination Press, 2015
* http://neuralnetworksanddeeplearning.com/

* This program creates a basic two-layer neural network that performs
* a simple three-input XOR function. Datasets must be in .csv format
* and consist of three columns of binary values and a forth column that
* is the XOR of the three values in the three input columns
'''

import numpy as np

# sigmoid activation function
def sigmoid(x):
	return 1.0/(1 + np.exp(-x))

# derivative of the sigmoid activation function
def sigmoid_prime(x):
	return sigmoid(x)*(1 - sigmoid(x))

# creates an identity vector of the desired output
def vector(j):
	e = np.zeros((2, 1))
	e[j] = 1.0
	return e

# loads input dataset and test dataset, and reshapes them
def data_loader(d_file, t_file=None):

	dataset = np.loadtxt(d_file, dtype=int, delimiter=";")
	x = dataset[:,0:3]
	y = dataset[:,3]

	td = (x, y)
	in_data = [np.reshape(i, (3, 1)) for i in td[0]]
	out_data = [vector(j) for j in td[1]]

	data = zip(in_data, out_data)

	if t_file:	
		testset = np.loadtxt(t_file, dtype=int, delimiter=";")
		f = testset[:,0:3]
		g = testset[:,3]

		ts = (f, g)
		in_test = [np.reshape(i, (3, 1)) for i in ts[0]]

		test = zip(in_test, ts[1])

		return(data, test)

	else: return(data)

'''
* This is the main class of the program. It creates a NeuralNet object instance when
* called and initialises the NN’s weights and biases arrays in the __init__ block.
* It has a number of methods within the class pertaining to the training of the NN
'''
class NeuralNet(object):

	def __init__(self, i, l, o):

		self.sizes = [i, l, o]
		self.layers = len(self.sizes)
		self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
		self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

	'''
	* Performs the feedforward algorithm during testing and prediction in the NeuralNet.
	* Returns the activation value a.
	'''
	def forwardprop(self, a):

		for b, w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w, a) + b)

		return a

	'''
	* This method runs most of the program. It trains the network on the input data using
	* Stochastic Gradient Descent (SGD). It will print out progress to the console, and
	* if test data is provided, then it will also print out performance metrics for each
	* training epoch.
	'''
	def train(self, data, epochs, batch, rate, tst=None):
		
		n = len(data)

		if tst:
			n2 = len(tst)

		for i in range(epochs):
			batches = [data[j:j+batch] for j in range(0, n, batch)]

			for x in batches:
				b_prime = [np.zeros(b.shape) for b in self.biases]
				w_prime = [np.zeros(w.shape) for w in self.weights]

				for u, v in x:
					db_p, dw_p = self.backprop(u, v)

					b_prime = [db+bx for db, bx in zip(b_prime, db_p)]
					w_prime = [dw+wx for dw, wx in zip(w_prime, dw_p)]

				self.weights = [w-(rate/len(x))*dw for w, dw in zip(self.weights, w_prime)]
				self.biases = [b-(rate/len(x))*db for b, db in zip(self.biases, b_prime)]

				correct = self.eval(tst)

			if tst:
				print("Epoch{:4d}:{:5d}/{} correct  |  Accuracy: {:5.2f}%".format(i+1, correct, n2, (correct/n2)*100.0))
			else:
				print("Epoch {0}/{1} complete".format(i+1, epochs))

	'''
	* Performs all the backpropagation operations on the NeuralNet during training.
	* Returns the derivative matrices for the biases and weights, which represent
	* the gradient for the cost function.
	'''
	def backprop(self, x, y):
		
		b_prime = [np.zeros(b.shape) for b in self.biases]
		w_prime = [np.zeros(w.shape) for w in self.weights]
		
		a = x
		activations = [x] # list to store all the activations, layer by layer
		z_vectors = [] # list to store all the z vectors, layer by layer

		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, a)+b
			z_vectors.append(z)
			a = sigmoid(z)
			activations.append(a)

		# backward pass
		delta = self.cost(activations[-1], y) * sigmoid_prime(z_vectors[-1])
		b_prime[-1] = delta
		w_prime[-1] = np.dot(delta, activations[-2].T)

		for l in range(2, self.layers):
			z = z_vectors[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].T, delta) * sp
			b_prime[-l] = delta
			w_prime[-l] = np.dot(delta, activations[-l-1].T)
		return (b_prime, w_prime)

	# Returns to partial derivative vector of the cost function and the activation a.
	def cost(self, x, y):
		return(x-y)

	# This method evaluates the network performance by returning the number of test
	# inputs that were correctly predicted in testing.
	def eval(self, tst):
		results = [(np.argmax(self.forwardprop(x)), y) for (x, y) in tst]

		return sum(int(x == y) for (x, y) in results)

# called when the program is run
if __name__ == "__main__":

	# User input
	trn_data = input("\nEnter the training data filename: ")
	tst_data = input("Enter the test data filename (if none, just press enter): ")
	epochs = int(input("Enter the number of training epochs: "))
	batch_size = int(input("Enter the batch size: "))
	learn_rate = float(input("Enter the learning rate (between 1.0 - 3.0): "))

	if tst_data:
		data, test = data_loader(trn_data, t_file=tst_data)
		test = list(test)
	else:
		data = data_loader(trn_data, t_file=None)

	data = list(data)

	net = NeuralNet(3, 4, 2)

	print()

	if tst_data:
		net.train(data, epochs, batch_size, learn_rate, tst=test)
	else: 
		net.train(data, epochs, batch_size, learn_rate, tst=None)
