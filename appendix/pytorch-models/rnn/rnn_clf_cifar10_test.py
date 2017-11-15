from rnn_clf import RNNClassifier
import tensorflow as tf


n_in = 32
cell_size = 128
n_layer = 1
n_out = 10
batch_size = 128
n_epoch = 5


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.cifar10.load_data()
    X_train = (X_train / 255.0).mean(axis=3)
    X_test = (X_test / 255.0).mean(axis=3)
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    rnn = RNNClassifier(n_in, n_out, cell_size, n_layer)
    rnn.fit(X_train, y_train, n_epoch, batch_size)
    rnn.evaluate(X_test, y_test, batch_size)
