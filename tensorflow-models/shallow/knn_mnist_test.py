import tensorflow as tf
from knn import KNN


def main():
    (X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.mnist.load_data()
    X_train = (X_train / 255.).reshape(-1, 28*28)
    X_test = (X_test / 255.).reshape(-1, 28*28)

    model = KNN(28*28)
    y_pred = model.predict(X_train, y_train, X_test)
    print("final testing accuracy: %.4f" % (y_pred == y_test).mean())


if __name__ == '__main__':
    main()