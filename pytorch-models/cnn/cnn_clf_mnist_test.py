from cnn_clf import CNNClassifier
import tensorflow as tf


img_size = (28,28)
img_ch = 1
kernel_size = 5
pool_size = 2
n_out = 10
batch_size = 128
n_epoch = 1


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.mnist.load_data()
    X_train = (X_train / 255.0).reshape(-1, img_ch, img_size[0], img_size[1])
    X_test = (X_test / 255.0).reshape(-1, img_ch, img_size[0], img_size[1])

    cnn = CNNClassifier(img_size=img_size, img_ch=img_ch, kernel_size=kernel_size, pool_size=pool_size, n_out=n_out)
    cnn.fit(X_train, y_train, n_epoch, batch_size)
    cnn.evaluate(X_test, y_test, batch_size)

