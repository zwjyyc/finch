import os, sys
import numpy as np
import project_modules
import cPickle as pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from data.processing import ChiWordMapper
from model.keras_based import build_keras_lstm


if __name__ == '__main__':
    # mapper = ChiWordMapper()
    # address = mapper.save_data()
    with open(os.path.join( os.path.dirname(os.getcwd()), 'data', 'Xy.pkl' ), 'rb') as input:
        X, y = pickle.load(input)
    X = np.array(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y)
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)

    model = build_keras_lstm(input_size=X.shape[2], time_steps=X.shape[1], cell_size=128, output_size=2)
    log = model.fit(X_train, y_train, nb_epoch=10, validation_data=(X_test, y_test), verbose=2)

    sns.set(context='poster', style='white')
    plt.plot(log.history['loss'], label='train_loss')
    plt.plot(log.history['acc'], label='train_acc')
    plt.plot(log.history['val_loss'], label='test_loss')
    plt.plot(log.history['val_acc'], label='test_acc')
    plt.legend(loc='best')
    plt.savefig("./log/" + sys.argv[0][:-3])
    print("Figure created !")

