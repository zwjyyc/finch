from autoencoder import Autoencoder
from keras.datasets import mnist
import matplotlib.pyplot as plt
import tensorflow as tf


if __name__ == '__main__':
    sess = tf.Session()
    auto = Autoencoder(sess, 28*28, [128,64,10,2], [2,10,64,128])
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train/255.0).reshape(-1, 28*28)
    X_test = (X_test/255.0).reshape(-1, 28*28)
    
    X_test_2d = auto.fit_transform(X_test)
    plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test)
    plt.show()
