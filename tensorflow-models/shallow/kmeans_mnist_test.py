import tensorflow as tf
from kmeans import KMeans


def main():
    (X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.mnist.load_data()
    X_train = (X_train / 255.).reshape(-1, 28*28)
    X_test = (X_test / 255.).reshape(-1, 28*28)
    Y_train = tf.contrib.keras.utils.to_categorical(y_train)
    print("Data Loaded")

    model = KMeans(k=25, n_features=28*28, n_classes=10)
    model.fit(X_train, Y_train)
    print("final testing accuracy: %.4f" % (model.predict(X_test) == y_test).mean())


if __name__ == '__main__':
    main()