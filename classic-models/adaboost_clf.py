import numpy as np
from sklearn.tree import DecisionTreeClassifier


class Adaboost:
    def __init__(self, n_models=100):
        self.n_models = n_models
        self.models = []
        self.alphas = []


    def fit(self, X, y):
        N = X.shape[0]
        W = np.ones(N) / N 
        for _ in range(self.n_models):
            tree = DecisionTreeClassifier(max_depth=1)
            tree.fit(X, y, sample_weight=W)
            preds = tree.predict(X)

            err = np.dot(W, preds != y)
            alpha = 0.5 * (np.log(1 - err) - np.log(err))

            W = W * np.exp(-alpha * y * preds)
            W = W / W.sum()

            self.models.append(tree)
            self.alphas.append(alpha)


    def predict(self, X):
        fx = np.zeros(X.shape[0])
        for alpha, model in zip(self.alphas, self.models):
            fx += alpha * model.predict(X)
        return np.sign(fx)


    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)
# end class