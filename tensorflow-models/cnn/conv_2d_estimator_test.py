from conv_2d_estimator import Estimator
import tensorflow as tf
import numpy as np


def main():
    (X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.mnist.load_data()
    X_train = (X_train / 255.)[:, :, :, np.newaxis].astype(np.float32)
    X_test = (X_test / 255.)[:, :, :, np.newaxis].astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    model = Estimator(10)
    model.fit(X_train, y_train)
    model.score(X_test, y_test)


if __name__ == '__main__':
    main()