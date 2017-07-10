from dcgan import Conv_GAN
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2


n_epoch = 4
batch_size = 32
G_size = 100


def gen_batch(arr, batch_size):
    for i in range(0, len(arr)-len(arr)%batch_size, batch_size):
        yield arr[i : i + batch_size]


if __name__ == '__main__':
    (X_train, y_train), (_, _) = tf.contrib.keras.datasets.mnist.load_data()
    X = np.expand_dims((X_train / 255.), 3)[y_train == 7]
    
    model = Conv_GAN(G_size, batch_size)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    plt.figure()
    for epoch in range(n_epoch):
        X = shuffle(X)
        for step, X_batch in enumerate(gen_batch(X, batch_size)):
            for _ in range(2):
                sess.run(model.G_train, {model.train_flag: True})
            sess.run(model.D_train, {model.X_in: X_batch, model.train_flag: True})
            G_loss, D_loss, D_prob, G_prob, loss = sess.run([model.G_loss, model.D_loss,
                                                             model.X_prob, model.G_prob,
                                                             model.mse],
                                                            {model.X_in: X_batch,
                                                             model.train_flag: False})
            print("Epoch %d/%d | Step %d/%d" % (epoch+1, n_epoch, step, int(len(X)/batch_size)))
            print("G loss: %.4f | D loss: %.4f | D prob: %.4f | G prob: %.4f | mse: %.4f" %
                 (G_loss, D_loss, D_prob.mean(), G_prob.mean(), loss))
    
        img = sess.run(model.G_out, {model.train_flag: False})[0]
        plt.subplot(2, 2, epoch+1)
        plt.imshow(np.squeeze(img))
    plt.tight_layout()
    plt.show()
    