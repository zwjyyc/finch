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
    clf = HighwayConvClassifier(img_size=(28,28), img_ch=1, pool_size=2, n_out=10, sess=sess)
    log = clf.fit(X_train, Y_train, n_epoch=10, keep_prob=0.5, val_data=(X_test,Y_test), en_exp_decay=True)
    pred = clf.predict(X_test)
    tf.reset_default_graph()
    final_acc = np.equal(np.argmax(pred,1), np.argmax(Y_test,1)).astype(float).mean()
    print("final testing accuracy: %.4f" % final_acc)
