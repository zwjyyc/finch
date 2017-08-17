from cnn_rnn_text_clf import ConvLSTMClassifier
import tensorflow as tf


vocab_size = 20000


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.imdb.load_data(num_words=vocab_size)

    clf = ConvLSTMClassifier(vocab_size)
    clf.fit(X_train, y_train, n_epoch=2)
    clf.evaluate(X_test, y_test)
