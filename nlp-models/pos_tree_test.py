import pos
import numpy as np
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = pos.load_data()
    X_train = np.expand_dims(x_train, 1)
    X_test = np.expand_dims(x_test, 1)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    print(model.score(X_train, y_train))
    print(model.score(X_test, y_test))
