from conv_rnn_text_clf import ConvLSTMClassifier
import tensorflow as tf
import numpy as np


max_features = 20000
maxlen = 100  # cut texts after this number of words (among top max_features most common words)
n_out = 2


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.imdb.load_data(num_words=max_features)

    X_train = tf.contrib.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = tf.contrib.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=maxlen)
    print('x_train shape:', X_train.shape, 'x_test shape:', X_test.shape)
    
    Y_train = tf.contrib.keras.utils.to_categorical(y_train)
    Y_test = tf.contrib.keras.utils.to_categorical(y_test)

    clf = ConvLSTMClassifier(maxlen, max_features, n_out)
    log = clf.fit(X_train, Y_train, batch_size=32, n_epoch=2, val_data=(X_test,Y_test))
    pred = clf.predict(X_test)

    final_acc = np.equal(np.argmax(pred,1), np.argmax(Y_test,1)).astype(float).mean()
    print("final testing accuracy: %.4f" % final_acc)
