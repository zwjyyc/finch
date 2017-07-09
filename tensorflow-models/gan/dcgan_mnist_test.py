from dcgan import Conv_GAN
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2


n_epoch = 5
batch_size = 32
G_size = 100


def gen_batch(arr, batch_size):
    for i in range(0, len(arr), batch_size):
        yield arr[i : i + batch_size]


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.mnist.load_data()
    X_train = X_train / 255.
    X = [cv2.resize(img, (64, 64)) for img in X_train]
    X = np.expand_dims(np.array(X), 3)
    X = X[y_train==3]
    
    model = Conv_GAN(G_size)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(n_epoch):
        X = shuffle(X)
        for step, real_data in enumerate(gen_batch(X, batch_size)):
            rand_data = np.random.randn(len(real_data), G_size)
            
            for _ in range(2):
                sess.run(model.G_train, {model.G_in: rand_data, model.train_flag: True})
            
            sess.run(model.D_train, {model.G_in: rand_data, model.X_in: real_data, model.train_flag: True})

            G_loss, D_loss, D_prob, G_prob, loss = sess.run([model.G_loss, model.D_loss,
                                                             model.X_true_prob, model.G_true_prob,
                                                             model.mse],
                                                            {model.G_in: rand_data,
                                                             model.X_in: real_data,
                                                             model.train_flag: False})
            
            print("Epoch %d/%d | Step %d/%d" % (epoch+1, n_epoch, step, int(len(X)/batch_size)))
            print("G loss: %.4f | D loss: %.4f | D prob: %.4f | G prob: %.4f | mse: %.4f " %
                 (G_loss, D_loss, D_prob.mean(), G_prob.mean(), loss))
    
    img = sess.run(model.G_out, {model.G_in: np.random.randn(1, G_size),
                                 model.train_flag: False})
    plt.imshow(np.squeeze(img))
    plt.savefig('./mnist')
    