from rnn_clf import RNNClassifier
import numpy as np
import tensorflow as tf


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train / 255.
    X_test = X_test / 255.

    clf = RNNClassifier(n_in=28, n_out=10, stateful=True)
    log = clf.fit(X_train, y_train, keep_prob_tuple=(0.8,1.0), val_data=(X_test, y_test))
    pred = clf.predict(X_test)

    final_acc = (pred == y_test).mean()
    print("final testing accuracy: %.4f" % final_acc)
