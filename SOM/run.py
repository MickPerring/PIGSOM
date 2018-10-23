import SOM as sm
import os.path
import sys

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

data = sm.data_loader(filename)

vector_length = len(data[0])-1
num_inputs = len(data)

# create SOM object
som = sm.SOM(map_width, map_height, vector_length, epochs, learning_rate, num_inputs)

# run program on SOM object
sm.run(som, data, epochs)