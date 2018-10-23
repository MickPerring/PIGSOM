import NeuralNetwork as Net

trn_data = "train_medium.csv"
tst_data = "test_medium.csv"
epochs = 50
batch_size = 10
learn_rate = 2.0

print("\nTraining file: {}\nTest file: {}\nEpochs: {}\nBatch size: {}\nLearning rate: {}".format(
	trn_data, tst_data, epochs, batch_size, learn_rate))

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
