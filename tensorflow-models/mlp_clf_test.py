from keras.datasets import mnist
from utils import to_one_hot
from mlp_clf import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os, sys


def plot(log, dir='./log'):
    if not os.path.exists(dir):
        os.makedirs(dir)
    sns.set(style='white')
    plt.plot(log['loss'], label='train_loss')
    plt.plot(log['val_loss'], label='test_loss')
    plt.legend(loc='best')
    plt.savefig(os.path.join(dir, sys.argv[0][:-3]))
    print("Figure created !")


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train / 255.0).reshape(-1, 28*28)
    X_test = (X_test / 255.0).reshape(-1, 28*28)
    y_train = to_one_hot(y_train)
    y_test = to_one_hot(y_test)

    clf = MLPClassifier(n_in=28*28, hidden_unit_list=[100, 200, 100], n_out=10)
    log = clf.fit(X_train, y_train, batch_size=100, en_exp_decay=True, val_data=(X_test, y_test))
    pred = clf.predict(X_test)
    clf.close()
    final_acc = np.equal(np.argmax(pred, 1), np.argmax(y_test, 1)).astype(float).mean()
    print("final testing accuracy: %.4f" % final_acc)

    plot(log)
