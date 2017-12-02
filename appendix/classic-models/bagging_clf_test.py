from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from bagging_clf import BaggingClassifier


if __name__ == '__main__':
    X, y = make_classification()
    
    tree = DecisionTreeClassifier()
    tree.fit(X, y)
    print ("score for tree model:", tree.score(X, y))
    
    clf = BaggingClassifier()
    clf.fit(X, y)
    print ("score for bagged model:", clf.score(X, y))
