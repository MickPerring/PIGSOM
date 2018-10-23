# PIGSOM

## Analysis of neural networks, self-organising maps and partial-input self-organising maps

### There are four different programs to run in here.

**Basic Neural Network (Neural_Net):**
- Open terminal and navigate to the Neural_Net directory
- To run program with user input parameters: > python run.py
  - _Input dataset must be specific format, see NeuralNetwork.py description at top of file_
- To run program on small dataset and default parameters: > python run_small.py
  - _train_small.csv and test_small.csv must be included in directory_
- To run program on medium dataset and default parameters: > python run_medium.py
  - _train_medium.csv and test_medium.csv must be included in directory_
- To run program on large dataset and default parameters: > python run_large.py
  - _train_large.csv and test_large.csv must be included in directory_

**Kears Neural Network (Keras_Net):**
- _Make sure that the Keras development environment is set up! See: https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/_
- Open Anaconda terminal and navigate to the Keras_Neural_Net directory
  - _pima-indians-diabetes.csv dataset file must be included in directory_
- To run program: > python run.py

**Self-Organising Map (SOM):**
- Open terminal and navigate to the SOM directory
- To run program with user input parameters: > python run.py
  - _Ensure dataset is compatible, see SOM.py description at top of file_

**Partial-Input Self-Organising Map (Partial_Input_SOM):**
- Open terminal and navigate to the Partial_Input_SOM directory
- To run program with user input parameters: > python run.py
  - _Ensure dataset is compatible, see PI_SOM.py description at top of file_
