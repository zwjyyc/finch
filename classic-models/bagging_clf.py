import numpy as np
from sklearn.tree import DecisionTreeClassifier


class BaggingClassifier:
    def __init__(self, model=DecisionTreeClassifier(), n_models=200):
        self.model = model
        self.n_models = n_models
        self.models = []
    def fit(self, X, y):
        for _ in range(self.n_models):
            N = len(X)
            idx = np.random.choice(N, size=N, replace=True)
            self.models.append(self.model.fit(X[idx], y[idx]))
    def predict(self, X):
        preds = np.zeros(len(X))
        for model in self.models:
            preds += model.predict(X)
        return np.round(preds / self.n_models)
    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(y == preds)
# end class