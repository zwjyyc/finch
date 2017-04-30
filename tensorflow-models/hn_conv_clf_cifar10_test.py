from keras.utils.np_utils import to_categorical as to_one_hot
from keras.datasets import cifar10
from hn_conv_clf import HighwayConvClassifier
import numpy as np
import tensorflow as tf


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    y_train = to_one_hot(y_train)
    y_test = to_one_hot(y_test)

    sess = tf.Session()
    clf = HighwayConvClassifier(img_size=(32,32), img_ch=3, pool_size=2, n_out=10, sess=sess)
    log = clf.fit(X_train, y_train, n_epoch=10, keep_prob=0.5, val_data=(X_test,y_test),
                  en_exp_decay=True)
    pred = clf.predict(X_test)
    tf.reset_default_graph()
    final_acc = np.equal(np.argmax(pred,1), np.argmax(y_test,1)).astype(float).mean()
    print("final testing accuracy: %.4f" % final_acc)
