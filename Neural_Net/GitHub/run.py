import NeuralNetwork as Net

# user input
trn_data = input("\nEnter the training data filename: ")
tst_data = input("Enter the test data filename (if none, just press enter): ")
epochs = int(input("Enter the number of training epochs: "))
batch_size = int(input("Enter the batch size: "))
learn_rate = float(input("Enter the learning rate (between 1.0 - 3.0): "))

if tst_data:
	data, test = Net.data_loader(trn_data, t_file=tst_data)
	test = list(test)
else:
	data = Net.data_loader(trn_data, t_file=None)

data = list(data)

net = Net.NeuralNet(3, 4, 2)

print()

if tst_data:
	net.train(data, epochs, batch_size, learn_rate, tst=test)
else: 
	net.train(data, epochs, batch_size, learn_rate, tst=None)