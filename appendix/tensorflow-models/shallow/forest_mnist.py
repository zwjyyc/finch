import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)


def main():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = (X_train / 255.).reshape(-1, 28*28).astype(np.float32)
    X_test = (X_test / 255.).reshape(-1, 28*28).astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
        num_classes=10, num_features=28*28, regression=False, num_trees=100, max_nodes=1000)
    estimator = tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator(params)

    estimator.fit(input_fn=tf.estimator.inputs.numpy_input_fn(
        x={'_':X_train}, y=y_train, batch_size=1000, num_epochs=10, shuffle=True))
    estimator.evaluate(input_fn=tf.estimator.inputs.numpy_input_fn(
        x={'_':X_test}, y=y_test, batch_size=1000, shuffle=False))


if __name__ == '__main__':
    main()