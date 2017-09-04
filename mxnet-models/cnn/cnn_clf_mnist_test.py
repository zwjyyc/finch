import tensorflow as tf
import numpy as np
from cnn_clf import CNNClassifier


def main():
    (X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.mnist.load_data()
    X_train = (X_train / 255.0)[:, np.newaxis, :, :]
    model = CNNClassifier(n_out=10)
    model.fit(X_train, y_train)


if __name__ == "__main__":
    main()
