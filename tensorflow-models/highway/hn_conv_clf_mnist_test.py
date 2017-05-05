from keras.datasets import mnist
from keras.utils.np_utils import to_categorical as to_one_hot
from hn_conv_clf import HighwayConvClassifier
import numpy as np
import tensorflow as tf


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train / 255.0)[:, :, :, np.newaxis]
    X_test = (X_test / 255.0)[:, :, :, np.newaxis]
    Y_train = to_one_hot(y_train)
    Y_test = to_one_hot(y_test)

    sess = tf.Session()
    clf = HighwayConvClassifier(sess, (28,28), 1, 10)
    log = clf.fit(X_train, Y_train, val_data=(X_test,Y_test))
    pred = clf.predict(X_test)
    tf.reset_default_graph()
    final_acc = np.equal(np.argmax(pred,1), np.argmax(Y_test,1)).astype(float).mean()
    print("final testing accuracy: %.4f" % final_acc)
