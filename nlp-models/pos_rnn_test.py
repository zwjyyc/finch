import pos
import numpy as np
import tensorflow as tf


SEQ_LEN = 5


def x_to_seq(*args):
    data = []
    for x in args:
        x = x[: (len(x) - len(x) % SEQ_LEN)]
        data.append(np.reshape(x, [-1, SEQ_LEN, 1]))
    return data


def y_to_seq(*args):
    data = []
    for y in args:
        y = y[: (len(y) - len(y) % SEQ_LEN)]
        n_class = np.max(y) + 1
        y = tf.contrib.keras.utils.to_categorical(y)
        data.append(np.reshape(y, [-1, SEQ_LEN, n_class]))
    return data


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = pos.load_data()
    X_train, X_test = x_to_seq(x_train, x_test)
    Y_train, Y_test = y_to_seq(y_train, y_test)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    