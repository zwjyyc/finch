from libsvm_clf import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np


if __name__ == '__main__':
    X, y = make_classification(5000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    svc = SVC()
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)

    acc = np.equal(y_pred, y_test).astype(float).mean()
    print('The testing accuracy is %.3f' % acc)
