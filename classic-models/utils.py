import numpy as np
import matplotlib.pyplot as plt


def make_data(N=500, D=2, n_center=4):
    X = np.random.randn(N, D)
    if n_center == 2:
        sep = 1.5
        X[:int(N/2)] += np.array([sep, sep])
        X[int(N/2):] += np.array([-sep, -sep])
        Y = np.array([0]*int(N/2) + [1]*int(N/2))
    if n_center == 4:
        sep = 2
        X[:125] += np.array([sep, sep])
        X[125:250] += np.array([sep, -sep])
        X[250:375] += np.array([-sep, -sep])
        X[375:] += np.array([-sep, sep])
        Y = np.array([0]*125 + [1]*125 + [0]*125 + [1]*125)
    return X, Y


def plot_decision_boundary(X, model):
    # step size in the mesh
    h = .02  
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max] x [y_min, y_max].
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
