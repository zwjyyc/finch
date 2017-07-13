from dcgan import DCGAN
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


n_epoch = 4
batch_size = 32
G_size = 100


def gen_batch(arr, batch_size):
    for i in range(0, len(arr), batch_size):
        yield arr[i : i + batch_size]


if __name__ == '__main__':
    (X_train, y_train), (_, _) = tf.contrib.keras.datasets.mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X = np.expand_dims(X_train, 3)[y_train == 6]
    
    gan = DCGAN(G_size)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    plt.figure()
    for epoch in range(n_epoch):
        X = shuffle(X)
        for step, images in enumerate(gen_batch(X, batch_size)):
            noise = np.random.randn(len(images), G_size)
            preds = sess.run(gan.G_out, {gan.G_in: noise, gan.train_flag: False})
            X_in = np.concatenate([images, preds])

            for _ in range(2):
                sess.run(gan.G_train, {gan.G_in: noise, gan.train_flag: True})
            sess.run(gan.D_train, {gan.G_in: np.concatenate([noise, noise]), gan.X_in: X_in, gan.train_flag: True})

            G_loss, D_loss, D_prob, G_prob, loss = sess.run([gan.G_loss, gan.D_loss,
                                                             gan.X_prob, gan.G_prob,
                                                             gan.mse],
                                                            {gan.G_in: noise,
                                                             gan.X_in: images,
                                                             gan.train_flag: False})
            print("Epoch %d/%d | Step %d/%d" % (epoch+1, n_epoch, step, int(len(X)/batch_size)))
            print("G loss: %.4f | D loss: %.4f | D prob: %.4f | G prob: %.4f | mse: %.4f" %
                 (G_loss, D_loss, D_prob.mean(), G_prob.mean(), loss))

        img = sess.run(gan.G_out, {gan.G_in: noise, gan.train_flag: False})[0]
        plt.subplot(2, 2, epoch+1)
        plt.imshow(np.squeeze(img))
    plt.tight_layout()
    plt.show()
    