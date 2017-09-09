import tensorflow as tf
from forest import Forest


def main():
    (X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.mnist.load_data()
    X_train = (X_train / 255.).reshape(-1, 28*28)
    X_test = (X_test / 255.).reshape(-1, 28*28)

    forest = Forest(28*28, 10)
    forest.fit(X_train, y_train)
    print("final testing accuracy: %.4f" % (forest.predict(X_test) == y_test).mean())


if __name__ == '__main__':
    main()