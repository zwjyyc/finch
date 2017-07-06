from cnn_rnn_text_clf import ConvLSTMClassifier
import tensorflow as tf


vocab_size = 20000
seq_len = 100


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.imdb.load_data(num_words=vocab_size)
    X_train = tf.contrib.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=seq_len)
    X_test = tf.contrib.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=seq_len)

    clf = ConvLSTMClassifier(vocab_size)
    clf.fit(X_train, y_train, n_epoch=1)
    clf.evaluate(X_test, y_test)
