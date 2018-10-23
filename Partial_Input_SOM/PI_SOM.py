'''
* Full PI_SOM.py program

* Author: Mick Perring
* Date: October 2018

* This program creates a Partial-Input Self-Organising Map based
* on user input parameters for the map dimentions, learning rate
* and number of training epochs. The user also chooses the input
* dataset on which to train the SOM. If dataset contains missing
* inputs, the program replaces them with zero
 
* SOM works on numerical datasets only, which include a class for
* each data sample. The class does not have to be numerical.

* Input datasets must be in .csv format, with each row a new data
* sample, and the final column containing the class of each sample
'''

import numpy as np
import os.path
import sys

# nice way to force floats to be printed with two decimal places
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

# loads input dataset into an ndarray of tuples of each row/sample in the dataset
def data_loader(d_file):
	data_test = np.genfromtxt(d_file, dtype=None, delimiter=";", encoding=None)
	return(data_test)

'''
* This is the main method of the program. It runs everything after the accepting of
* user input, loading of the data file (using data_loader()) and the SOM initialisation
'''
def run(som, dataset, epochs):

	for epoch in range(epochs):
		np.random.shuffle(dataset) # reshuffle dataset afater each epoch

		d_type = dataset[0][len(dataset[0])-1].dtype # get the datatype of the classes

		# if the classes are strings, ensure that dtype is set to the maximum string
		# length of the classes
		if(d_type != int and d_type!= float):
			temp = dataset[0][len(dataset[0])-1]
			for i in range(len(dataset)):
				if(len(dataset[i][len(dataset[0])-1]) > len(temp)):
					temp = dataset[i][len(dataset[0])-1]		
			d_type = temp.dtype

		data = np.zeros((len(dataset),len(dataset[0])-1))
		classification = np.empty((len(dataset), 1), dtype=d_type)

		# split dataset into input data and classes
		for i in range(len(dataset)):
			for j in range(len(dataset[0])-1):
				data[i][j] = dataset[i][j]
			classification[i][0] = dataset[i][len(dataset[0])-1]

		# creates an array to count the number of winners in each class for each node
		if(epoch == 0):
			for j in classification:
				if(np.isin(j, som.classes, invert=True)):
					som.classes.append(j[0])
			som.classify_count = np.zeros((som.y, som.x, len(som.classes)), dtype=int)
			print("\nTraining:\n")

		som.epoch = epoch+1

		count = 0
		error = 0

		# iterate through each input vector in the dataset
		for in_vector in data:
			for i in range(som.input_dim):
				# partial-input function, replaces missing data entries with 0 
				if(np.isnan(in_vector[i])):
					in_vector[i] = 0
				if(epoch == 0):
					# track the maximum and minumum value for each attribute in the dataset
					if(in_vector[i] > som.data_max[i]):
						som.data_max[i] = in_vector[i]
					elif(in_vector[i] < som.data_min[i]):
						som.data_min[i] = in_vector[i]

			som.BMU(in_vector, classification[count])

			win_x = som.winner[0]
			win_y = som.winner[1]

			# track the accuracy of the input data to the weight of the winner
			win_vec = som.map[win_x][win_y]
			som.accuracy_vector += np.sqrt((win_vec - in_vector)**2)

			# index of the highest winning class at the node, determines node's class
			index = np.argmax(som.classify_count[win_x][win_y])
			
			# tracks prediction error
			if(classification[count] != som.classes[index]):
				error+=1

			som.neighbourhood_func(in_vector)
			som.sigma_calc()
			som.eta_calc()
			som.time+=1

			count+=1

		if (epoch == 0):
			som.data_range = som.data_max - som.data_min

		som.accuracy_vector /= len(data)

		# accuracy calculation
		accuracy = 100 - (np.sum((som.accuracy_vector/som.data_range)*100))/som.input_dim

		# error calculation
		error = (error/len(data))*100

		# output performance results for each epoch to the console
		print("Epoch:{:4d}/{}  |  Accuracy: {:5.2f}%  |  Error: {:5.2f}%".format(som.epoch, som.epochs, accuracy, error))

		# reset accuracy vector
		som.accuracy_vector = np.zeros(som.input_dim)

'''
* This is the main class of the program. It creates a SOM object instance
* when called and initialises all of the SOM object’s variables and arrays
* in the __init__ block. It has a number of methods within the class
* pertaining to the training of the SOM
'''
class SOM(object):

	# initialise all of the SOM's variables and arrays
	def __init__(self, latx, laty, dim, epchs, rate, num_inputs):

		self.x 					= latx
		self.y 					= laty
		self.input_dim 			= dim
		self.epochs 			= epchs
		self.epoch 				= epchs
		self.iterations 		= epchs*num_inputs
		self.time 				= 0
		self.radius 			= max(latx, laty)/2
		self.rate 				= rate
		self.n_sigma 			= self.radius
		self.n_eta 				= rate
		self.n_lambda 			= self.iterations/np.log(self.radius)
		self.n_theta 			= np.array([])
		self.euclidian_array	= np.array([])
		self.dist_array 		= np.zeros((laty, latx))
		self.classify_count		= [[[]]]
		self.classes			= []
		self.winner 			= ()
		self.winner_count 		= np.zeros((laty, latx), dtype=int)
		self.data_max 			= np.zeros(dim, dtype=float)
		self.data_min 			= np.zeros(dim, dtype=float)
		self.data_range 		= np.zeros(dim, dtype=float)
		self.accuracy_vector 	= np.zeros(dim)
		self.neighbours 		= []
		self.map 				= np.random.normal(5.0, 2.5, size=(laty, latx, dim))

	'''
	* Calculates the difference (Euclidian distance) between the in_vector (input vector)
	* and the weight of each of the SOM nodes, and then determines the BMU (winning node)
	* based on which node has the least difference between its weight and in_vector
	'''
	def BMU(self, in_vector, classification):
		self.euclidian_array = np.sum((self.map - in_vector)**2, axis=2)
		self.winner = np.unravel_index(np.argmin(self.euclidian_array), self.euclidian_array.shape)
		win_x = self.winner[0]
		win_y = self.winner[1]
		# count number of winners at each node
		self.winner_count[win_x][win_y] += 1
		
		# count number of winners for each classification
		for i in range(len(self.classes)):
			if(self.classes[i] == classification):
				self.classify_count[win_x][win_y][i] += 1

	'''
	* Finds all nodes within the neighbourhood region of the winner by calculating whether their
	* distance from the BMU is within the current neighbourhood radius (n_sigma)
	'''
	def neighbourhood_func(self, in_vector):
		self.neighbours = []

		for x in range(len(self.map)):
		    for y in range(len(self.map[x])):
			    diff_x, diff_y = np.subtract((x, y), (self.winner))
			    cartesian_dist = np.sqrt(diff_x**2 + diff_y**2)
			    self.dist_array[x][y] = cartesian_dist
			    if(cartesian_dist <= self.n_sigma):
			        self.neighbours.append((x,y))

		# neighbourhood function of the SOM, updates n_theta for each node in the neighbourhood
		self.theta_calc()

		for i in range(len(self.neighbours)):
			x = self.neighbours[i][0]
			y = self.neighbours[i][1]
			weight = self.map[x][y]
			theta = self.n_theta[x][y]

			# perform the weight adjustment for each nde in the neighbourhood of the winner,
			# including the winner itself
			self.map[x][y] = weight + (theta*self.n_eta*(in_vector - weight))

	# updates n_sigma (neighbourhood radius)
	def sigma_calc(self):
		self.n_sigma = self.radius*np.exp(-self.time/self.n_lambda)

	# updates n_eta (learning rate)
	def eta_calc(self):
		self.n_eta = self.rate*np.exp(-self.time/self.iterations)

	# updates n_theta array (neighbourhood function)
	def theta_calc(self):
		self.n_theta = np.exp(-(self.dist_array**2)/(2*self.n_sigma**2))
		
# called if the program is run directly, instead of from a run.py script
if __name__ == '__main__':

	exists = False

	# check if file exists, loops on incorrect filenames unless user types exit
	while (exists == False):
		filename = input("\nEnter filename: ")
		if(filename == 'exit'):
			sys.exit(0)
		elif(os.path.exists(filename)):
			exists = True
		else:
			print("\nFile not found. Please try again. To exit, type 'exit' and press return.")

	# user input for SOM variables
	map_width = int(input("Enter map width: "))
	map_height = int(input("Enter map height: "))
	epochs = int(input("Enter number of epochs: "))
	learning_rate = float(input("Enter learning rate: "))

	data = data_loader(filename)

	vector_length = len(data[0])-1
	num_inputs = len(data)

	# create SOM object
	som = SOM(map_width, map_height, vector_length, epochs, learning_rate, num_inputs)

	# run program on SOM object
	run(som, data, epochs)