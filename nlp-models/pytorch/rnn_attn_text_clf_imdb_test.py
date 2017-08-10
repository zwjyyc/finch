from rnn_attn_text_clf import RNNTextClassifier
import tensorflow as tf


vocab_size = 20000
maxlen = 250
n_epoch = 2


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.imdb.load_data(num_words=vocab_size)
    X_train = tf.contrib.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = tf.contrib.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=maxlen)

    clf = RNNTextClassifier(vocab_size)
    clf.fit(X_train, y_train, n_epoch=n_epoch)
    clf.evaluate(X_test, y_test)
