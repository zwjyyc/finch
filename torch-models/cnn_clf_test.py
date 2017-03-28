from cnn_clf import CNNClassifier
from keras.datasets import mnist

# Hyper Parameters
img_h = 28
img_w = 28
num_classes = 10
batch_size = 100
num_epochs = 2

# MNIST Dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train / 255.0).reshape(-1, 1, img_h, img_w)
X_test = (X_test / 255.0).reshape(-1, 1, img_h, img_w)
cnn = CNNClassifier(img_h, img_w, num_classes)
cnn.fit(X_train, y_train, num_epochs, batch_size)
cnn.evaluate(X_test, y_test, batch_size)

