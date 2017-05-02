from rnn_clf import RNNClassifier
from keras.datasets import mnist


n_in = 28
cell_size = 128
n_layer = 2
n_out = 10
batch_size = 100
n_epoch = 2


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    rnn = RNNClassifier(n_in, cell_size, n_layer, n_out)
    rnn.fit(X_train, y_train, n_epoch, batch_size)
    rnn.evaluate(X_test, y_test, batch_size)

