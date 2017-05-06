from keras.datasets import cifar10
from conv_ae import ConvAE
import matplotlib.pyplot as plt
import tensorflow as tf


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = (X_train/255.0).reshape(-1, 32, 32, 3)
    X_test = (X_test/255.0).reshape(-1, 32, 32, 3)

    sess = tf.Session()
    auto = ConvAE(sess, (32, 32), 3)
    X_test_pred = auto.fit_transform(X_test, n_epoch=3)
    
    plt.imshow(X_test[21].reshape(32,32,3))
    plt.show()
    plt.imshow(X_test_pred[21].reshape(32,32,3))
    plt.show()
