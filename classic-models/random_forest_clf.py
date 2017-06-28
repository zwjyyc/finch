from sklearn.tree import DecisionTreeClassifier
import math
import numpy as np
from utils import one_hot


class RandomForestClassifier:
    def __init__(self, n_trees=100):
        self.n_trees = n_trees
        self._forest = [DecisionTreeClassifier() for _ in range(n_trees)]
        self._features = []


    def fit(self, X, y):
        N = X.shape[0]
        D = X.shape[1]
        for tree in self._forest:
            N_ = np.random.choice(np.arange(N), size=int(math.sqrt(N)), replace=True)
            D_ = np.random.choice(np.arange(D), size=int(math.sqrt(D)), replace=True)
            tree.fit(X[N_][:, D_], y[N_])
            self._features.append(D_)
    

    def predict(self, X):
        ys = [tree.predict(X[:, self._features[i]]) for i, tree in enumerate(self._forest)]
        ys_one_hot = [one_hot(y) for y in ys]
        return np.argmax(np.sum(ys_one_hot, axis=0), axis=1)
    

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(y == preds)
