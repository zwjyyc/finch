from keras.datasets import cifar10
from conv_ae import Autoencoder
import matplotlib.pyplot as plt
import tensorflow as tf


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = (X_train/255.0).reshape(-1, 32, 32, 3)
    X_test = (X_test/255.0).reshape(-1, 32, 32, 3)

    sess = tf.Session()
    auto = Autoencoder(sess, (32, 32), 3)
    X_test_2d = auto.fit_transform(X_test)
    
    plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test)
    plt.show()
