from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical as to_one_hot
from conv_1d_text_clf import Conv1DClassifier
import tensorflow as tf


batch_size = 32
max_features = 5000
maxlen = 400  # cut texts after this number of words (among top max_features most common words)
embedding_dims = 50
n_filters = 250
kernel_size = 3
hidden_dims = 250


if __name__ == '__main__':
    print('Loading data...')
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    print('Pad sequences (samples x time)')
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    print('x_train shape:', X_train.shape)
    print('x_test shape:', X_test.shape)
    Y_train = to_one_hot(y_train)
    Y_test = to_one_hot(y_test)

    sess = tf.Session()
    clf = Conv1DClassifier(maxlen, max_features, embedding_dims,
                           n_filters, kernel_size, hidden_dims,
                           n_out=2, sess=sess)
    log = clf.fit(X_train, Y_train, keep_prob=0.8, val_data=(X_test,Y_test))
    pred = clf.predict(X_test)
    tf.reset_default_graph()

    final_acc = np.equal(np.argmax(pred,1), np.argmax(Y_test,1)).astype(float).mean()
    print("final testing accuracy: %.4f" % final_acc)
