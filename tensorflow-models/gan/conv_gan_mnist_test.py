from conv_gan import CONV_GAN
import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    gan = CONV_GAN((32, 32, 3))

    (X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.mnist.load_data()
    print("data loaded")
    X_train = (X_train / 255.0)[:, :, :, np.newaxis]
    X_test = (X_test / 255.0)[:, :, :, np.newaxis]

    gan.fit(X_train)
