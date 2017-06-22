from conv_1d_text_clf import Conv1DClassifier
import tensorflow as tf
import numpy as np


vocab_size = 5000
seq_len = 400  # cut texts after this number of words (among top max_features most common words)
n_out = 2


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.imdb.load_data(num_words=vocab_size)

    X_train = tf.contrib.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=seq_len)
    X_test = tf.contrib.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=seq_len)
    print('X_train shape:', X_train.shape, 'X_test shape:', X_test.shape)
    
    Y_train = tf.contrib.keras.utils.to_categorical(y_train)
    Y_test = tf.contrib.keras.utils.to_categorical(y_test)

    clf = Conv1DClassifier(seq_len, vocab_size, n_out)
    log = clf.fit(X_train, Y_train, n_epoch=3, batch_size=32, keep_prob=0.8, en_exp_decay=True,
                  val_data=(X_test,Y_test))
    pred = clf.predict(X_test)

    final_acc = (np.argmax(pred,1) == np.argmax(Y_test,1)).mean()
    print("final testing accuracy: %.4f" % final_acc)
