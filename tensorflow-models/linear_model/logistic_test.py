from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from logistic import Logistic
import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    X, y = make_classification(5000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    Y_train = tf.contrib.keras.utils.to_categorical(y_train)
    Y_test = tf.contrib.keras.utils.to_categorical(y_test)

    clf = Logistic(X.shape[1], 2)
    clf.fit(X_train, Y_train, n_epoch=100, val_data=(X_test, Y_test))
    Y_pred = clf.predict(X_test)
    final_acc = (np.argmax(Y_pred,1) == np.argmax(Y_test,1)).astype(float).mean()
    print("final testing accuracy: %.4f" % final_acc)

    clf = SVC(kernel='linear')
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    print("svm (sklearn):", (y_pred == y_test).astype(float).mean())
