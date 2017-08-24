from variational_ae import Autoencoder
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import torch


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.mnist.load_data()
    X_train = (X_train/255.0).reshape(-1, 28*28)
    X_test = (X_test/255.0).reshape(-1, 28*28)

    auto = Autoencoder(28*28, [500,500,20])
    auto.fit(X_train)

    original = torch.autograd.Variable(torch.from_numpy(np.expand_dims(X_test[41],0).astype(np.float32)))
    restored, _ = auto(original)
    plt.imshow(X_test[41].reshape(28,28))
    plt.show()
    plt.imshow(restored.data.numpy().reshape(28,28))
    plt.show()
    
