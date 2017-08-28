from dcgan import GAN
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn
import numpy as np
import torch


N_EPOCH = 15
BATCH_SIZE = 32
G_SIZE = 100


def gen_batch(arr, batch_size):
    for i in range(0, len(arr), batch_size):
        yield arr[i : i + batch_size]


def scaled(images):
    return ( images.astype(np.float32) - (255./2) ) / (255./2)


def select(images, labels, num):
    return np.expand_dims(images, 1)[labels == num]


if __name__ == '__main__':
    (X_train, y_train), (_, _) = tf.contrib.keras.datasets.mnist.load_data()
    X = select(scaled(X_train), y_train, 8)
    gan = GAN(G_SIZE, (28, 28), 1, shape_trace=[(7, 7, 128), (14, 14, 64)])
    
    plt.figure()
    for i, epoch in enumerate(range(N_EPOCH)):
        X = sklearn.utils.shuffle(X)
        for step, images in enumerate(gen_batch(X, BATCH_SIZE)):
            G_loss, D_loss, D_prob, G_prob, mse = gan.train_op(images)
            print("Epoch %d/%d | Step %d/%d" % (epoch+1, N_EPOCH, step, len(X)//BATCH_SIZE))
            print("G loss: %.4f | D loss: %.4f | D prob: %.4f | G prob: %.4f | mse: %.4f" %
                 (G_loss.data[0], D_loss.data[0], D_prob.data.mean(), G_prob.data.mean(), mse.data[0]))
        if i in range(N_EPOCH-4, N_EPOCH):
            gan.g.eval()
            img = gan.g(torch.autograd.Variable(torch.randn(1, G_SIZE)))
            plt.subplot(2, 2, i+1-(N_EPOCH-4))
            plt.imshow(np.squeeze(img.data.numpy()))
    plt.tight_layout()
    plt.show()
