from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from keras.utils.np_utils import to_categorical as to_one_hot
from logistic import Logistic
import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    X, y = make_classification(5000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    Y_train = to_one_hot(y_train)
    Y_test = to_one_hot(y_test)

    sess = tf.Session()
    clf = Logistic(l1_ratio=0.15, n_in=X.shape[1], n_out=2, sess=sess)
    clf.fit(X_train, Y_train, n_epoch=100, val_data=(X_test, Y_test))
    Y_pred = clf.predict(X_test)
    final_acc = np.equal(np.argmax(Y_pred,1), np.argmax(Y_test,1)).astype(float).mean()
    print("final testing accuracy: %.4f" % final_acc)

    clf = SVC(kernel='linear')
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    print("svm (sklearn):", np.equal(y_pred, y_test).astype(float).mean())
