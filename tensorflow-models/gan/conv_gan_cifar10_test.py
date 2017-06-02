from conv_gan import CONV_GAN
import tensorflow as tf


if __name__ == '__main__':
    gan = CONV_GAN((32, 32, 3))

    (X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.cifar10.load_data()
    print("data loaded")
    X_train = (X_train / 255.0)
    X_test = (X_test / 255.0)

    gan.fit(X_train)
