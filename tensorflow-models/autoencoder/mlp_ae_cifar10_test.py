from mlp_ae import Autoencoder
import matplotlib.pyplot as plt
import tensorflow as tf


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.cifar10.load_data()
    X_train = (X_train/255.0).mean(axis=3).reshape(-1, 32*32)
    X_test = (X_test/255.0).mean(axis=3).reshape(-1, 32*32)
    
    sess = tf.Session()
    auto = Autoencoder(sess, 32*32, [128,64,10,2], [2,10,64,128])
    auto.fit(X_train, val_data=X_test)
    X_test_2d = auto.transform(X_test)
    X_test_pred = auto.predict(X_test)
    
    plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test)
    plt.show()
    plt.imshow(X_test[21].reshape(32,32))
    plt.show()
    plt.imshow(X_test_pred[21].reshape(32,32))
    plt.show()
