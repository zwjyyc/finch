import pos
import numpy as np
import tensorflow as tf
from rnn_seq2seq_clf import RNNTextClassifier

SEQ_LEN = 5
BATCH_SIZE = 128

def x_to_seq(*args):
    data = []
    for x in args:
        x = x[: (len(x) - len(x) % SEQ_LEN)]
        data.append(np.reshape(x, [-1, SEQ_LEN]))
    return data


def y_to_seq(*args):
    data = []
    for y in args:
        y = y[: (len(y) - len(y) % SEQ_LEN)]
        y = tf.contrib.keras.utils.to_categorical(y)
        data.append(np.reshape(y, [-1, SEQ_LEN, n_class]))
    return data

if __name__ == '__main__':
    x_train, y_train, x_test, y_test, vocab_size, n_class = pos.load_data()
    X_train, X_test = x_to_seq(x_train, x_test)
    Y_train, Y_test = y_to_seq(y_train, y_test)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    clf = RNNTextClassifier(SEQ_LEN, vocab_size, n_class)
    clf.fit(X_train, Y_train, val_data=(X_test, Y_test), rnn_keep_prob=0.7, n_epoch=5, batch_size=BATCH_SIZE)
    Y_pred = clf.predict(X_test, batch_size=BATCH_SIZE)
    Y_test = Y_test.reshape(-1, n_class)
    final_acc = (np.argmax(Y_pred,1) == np.argmax(Y_test,1)).mean()
    print("final testing accuracy: %.4f" % final_acc)
