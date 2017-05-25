from rnn_text_clf import RNNTextClassifier
import tensorflow as tf
import numpy as np


vocab_size = 20000
maxlen = 80
batch_size = 32


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.imdb.load_data(num_words=vocab_size)

    X_train = tf.contrib.keras.preprocessing.sequence.pad_sequences(X_train, maxlen)
    X_test = tf.contrib.keras.preprocessing.sequence.pad_sequences(X_test, maxlen)
    print('X_train shape:', X_train.shape, '|', 'X_test shape:', X_test.shape)
    
    Y_train = tf.contrib.keras.utils.to_categorical(y_train)
    Y_test = tf.contrib.keras.utils.to_categorical(y_test)

    clf = RNNTextClassifier(maxlen, vocab_size, 2)
    log = clf.fit(X_train, Y_train, n_epoch=3, batch_size=batch_size, rnn_keep_prob=0.8, en_exp_decay=True,
                  val_data=(X_test, Y_test))
    Y_pred = clf.predict(X_test, batch_size)

    final_acc = np.equal(np.argmax(Y_pred,1), np.argmax(Y_test,1)).astype(float).mean()
    print("final testing accuracy: %.4f" % final_acc)
