from keras.utils.np_utils import to_categorical as to_one_hot
from keras.datasets import cifar10
from hn_mlp_clf import HighwayMLPClassifier
import numpy as np
import time
import tensorflow as tf


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = (X_train/255.0).mean(axis=3).reshape(-1, 32*32)
    X_test = (X_test/255.0).mean(axis=3).reshape(-1, 32*32)
    Y_train = to_one_hot(y_train)
    Y_test = to_one_hot(y_test)

    sess = tf.Session()
    clf = HighwayMLPClassifier(sess, n_in=32*32, n_hidden=16*16, n_highway=5, n_out=10)

    t0 = time.time()
    
    log = clf.fit(X_train, Y_train, val_data=(X_test,Y_test))
    pred = clf.predict(X_test)
    tf.reset_default_graph()
    final_acc = np.equal(np.argmax(pred,1), np.argmax(Y_test,1)).astype(float).mean()
    print("final testing accuracy: %.4f" % final_acc)

    print("total time:", time.time()-t0)
