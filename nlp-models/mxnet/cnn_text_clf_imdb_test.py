from cnn_text_clf import CNNTextClassifier
import tensorflow as tf
import mxnet as mx


vocab_size = 5000
maxlen = 400


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.imdb.load_data(num_words=vocab_size)
    X_train = tf.contrib.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = tf.contrib.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=maxlen)

    clf = CNNTextClassifier(mx.cpu(), vocab_size)
    clf.fit(X_train, y_train, val_data=(X_test, y_test))
