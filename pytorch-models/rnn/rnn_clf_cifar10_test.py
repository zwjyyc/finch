from rnn_clf import RNNClassifier
from keras.datasets import cifar10


n_in = 32
cell_size = 128
n_layer = 1
n_out = 10
batch_size = 128
n_epoch = 5


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = (X_train / 255.0).mean(axis=3)            # rbg averaging to grayscale
    X_test = (X_test / 255.0).mean(axis=3)              # rgb averaging to grayscale
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    rnn = RNNClassifier(n_in, cell_size, n_layer, n_out)
    rnn.fit(X_train, y_train, n_epoch, batch_size)
    rnn.evaluate(X_test, y_test, batch_size)
