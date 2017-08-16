from __future__ import print_function
from rnn_text_clf import RNNTextClassifier
import tensorflow as tf


vocab_size = 20000
n_epoch = 2


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.imdb.load_data(num_words=vocab_size)
    print("Data Loaded")

    clf = RNNTextClassifier(vocab_size)
    clf.fit(X_train, y_train, n_epoch=n_epoch)
    clf.evaluate(X_test, y_test)
