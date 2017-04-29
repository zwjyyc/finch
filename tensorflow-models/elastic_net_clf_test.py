from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from elastic_net_clf import ElasticNetClassifier
import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    X, y = make_classification(5000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    sess = tf.Session()
    clf = ElasticNetClassifier(l1_ratio=0.15, n_in=X.shape[1], n_out=1, sess=sess)
    clf.fit(X_train, y_train, val_data=(X_test, y_test))
    y_pred = clf.predict(X_test)
    final_acc = np.equal(np.argmax(y_pred,1), np.argmax(y_test,1)).astype(float).mean()
    print("final testing accuracy: %.4f" % final_acc)
