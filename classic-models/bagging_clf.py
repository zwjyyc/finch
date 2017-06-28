import numpy as np
from utils import one_hot
from sklearn.tree import DecisionTreeClassifier


class BaggingClassifier:
    def __init__(self, base_model=DecisionTreeClassifier(), n_models=10):
        self.base_model = base_model
        self.n_models = n_models
        self.models = []
    

    def fit(self, X, y):
        for _ in range(self.n_models):
            N = len(X)
            idx = np.random.choice(N, size=N, replace=True)
            self.models.append(self.base_model.fit(X[idx], y[idx]))
    

    def predict(self, X):
        ys = [model.predict(X) for model in self.models]
        ys_one_hot = [one_hot(y) for y in ys]
        return np.argmax(np.sum(ys_one_hot, axis=0), axis=1)
    
    
    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(y == preds)
# end class