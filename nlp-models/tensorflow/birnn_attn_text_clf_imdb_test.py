from birnn_attn_text_clf import BiRNNTextClassifier
import tensorflow as tf
import numpy as np


vocab_size = 20000
seq_len = 80
batch_size = 32


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.imdb.load_data(num_words=vocab_size)

    X_train = tf.contrib.keras.preprocessing.sequence.pad_sequences(X_train, seq_len)
    X_test = tf.contrib.keras.preprocessing.sequence.pad_sequences(X_test, seq_len)
    print('X_train shape:', X_train.shape, '|', 'X_test shape:', X_test.shape)
    
    clf = BiRNNTextClassifier(seq_len, vocab_size, 2)
    log = clf.fit(X_train, y_train, n_epoch=3, batch_size=batch_size, keep_prob=0.8, en_exp_decay=True,
                  val_data=(X_test, y_test))
    y_pred = clf.predict(X_test, batch_size)

    final_acc = (y_pred == y_test).mean()
    print("final testing accuracy: %.4f" % final_acc)
