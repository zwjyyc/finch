from mlp_clf import MLPClassifier
from keras.datasets import mnist

# Hyper Parameters
input_size = 28*28
hidden_1_size = 100
hidden_2_size = 200
hidden_3_size = 100
num_classes = 10
batch_size = 100
num_epochs = 5

# MNIST Dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)
mlp = MLPClassifier(input_size, hidden_1_size, hidden_2_size, hidden_3_size, num_classes)
mlp.fit(X_train, y_train, num_epochs, batch_size)
mlp.evaluate(X_test, y_test, batch_size)

