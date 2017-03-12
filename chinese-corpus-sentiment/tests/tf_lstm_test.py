import os, sys
import numpy as np
import project_modules
import cPickle as pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from data.processing import ChiWordMapper
from model.tf_based import RNNClassifier


if __name__ == '__main__':
    # mapper = ChiWordMapper()
    # address = mapper.save_data()
    with open(os.path.join( os.path.dirname(os.getcwd()), 'data', 'Xy.pkl' ), 'rb') as input:
        X, y = pickle.load(input)
    X = np.array(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y)

    model = RNNClassifier(n_in=X.shape[2], n_step=X.shape[1], n_hidden_units=128, n_out=2)
    log = model.fit(X_train, y_train, nb_epoch=20, validation_data=(X_test, y_test))

    sns.set(style='white')
    plt.plot(log['loss'], label='train_loss')
    plt.plot(log['acc'], label='train_acc')
    plt.plot(log['val_loss'], label='test_loss')
    plt.plot(log['val_acc'], label='test_acc')
    plt.legend(loc='best')
    plt.savefig("./log/" + sys.argv[0][:-3])
    print("Figure created !")

