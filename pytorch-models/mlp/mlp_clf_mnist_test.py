from mlp_clf import MLPClassifier
from keras.datasets import mnist


n_in = 28*28
hidden_units = [300, 200, 100]
n_out = 10
batch_size = 100
n_epoch = 5


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train / 255.0).reshape(-1, n_in)
    X_test = (X_test / 255.0).reshape(-1, n_in)

    mlp = MLPClassifier(n_in, hidden_units, n_out)
    mlp.fit(X_train, y_train, n_epoch, batch_size)
    mlp.evaluate(X_test, y_test, batch_size)

