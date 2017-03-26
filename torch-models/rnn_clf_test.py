from rnn_clf import RNNClassifier
from keras.datasets import mnist

# Hyper Parameters
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2

# MNIST Dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
rnn = RNNClassifier(input_size, hidden_size, num_layers, num_classes)
rnn.fit(X_train, y_train, num_epochs, batch_size)
rnn.evaluate(X_test, y_test, batch_size)

