from keras.datasets import mnist
from keras.utils.np_utils import to_categorical as to_one_hot
from mlp_clf import MLPClassifier
import numpy as np
import tensorflow as tf


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train/255.0).reshape(-1, 28*28)
    X_test = (X_test/255.0).reshape(-1, 28*28)
    Y_train = to_one_hot(y_train)
    Y_test = to_one_hot(y_test)

    sess = tf.Session()
    clf = MLPClassifier(n_in=28*28, hidden_unit_list=[300,200,100], n_out=10, sess=sess)
    log = clf.fit(X_train, Y_train, val_data=(X_test,Y_test))
    pred = clf.predict(X_test)
    tf.reset_default_graph()
    final_acc = np.equal(np.argmax(pred, 1), np.argmax(Y_test, 1)).astype(float).mean()
    print("final testing accuracy: %.4f" % final_acc)
