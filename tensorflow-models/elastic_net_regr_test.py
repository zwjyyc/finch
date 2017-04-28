from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import make_regression
from elastic_net_regr import ElasticNetRegression
import tensorflow as tf


if __name__ == '__main__':
    X, y = make_regression(5000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    sess = tf.Session()
    regr = ElasticNetRegression(l1_ratio=0.15, n_in=X.shape[1], sess=sess)
    regr.fit(X_train, y_train, val_data=(X_test, y_test))
    y_pred = regr.predict(X_test)
    print("R2: ", r2_score(y_pred, y_test))
