from utils import make_data
from utils import plot_decision_boundary
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


if __name__ == '__main__':
    X, Y = make_data()
    plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
    
    tree = DecisionTreeClassifier()
    tree.fit(X, Y)
    
    plot_decision_boundary(X, tree)
    plt.show()
