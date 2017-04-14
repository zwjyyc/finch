import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def to_one_hot(y, num_classes=None):
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    one_hot = np.zeros((n, num_classes))
    one_hot[np.arange(n), y] = 1
    return one_hot


def load_mnist():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    return mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
