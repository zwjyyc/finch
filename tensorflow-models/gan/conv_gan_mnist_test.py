from conv_gan_mnist import ConvGAN
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


N_EPOCH = 10
BATCH_SIZE = 128


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.mnist.load_data()
    X_train = (X_train / 255.0).reshape(-1, 28, 28, 1)
    X_test = (X_test / 255.0).reshape(-1, 28, 28, 1)
    
    model = ConvGAN((28, 28, 1))
    
    n_batch = len(X_train) / BATCH_SIZE
    model.sess.run(tf.global_variables_initializer())
    
    for epoch in range(N_EPOCH):
        X = shuffle(X_train)
        for local_step, X_batch in enumerate(model.gen_batch(X, BATCH_SIZE)):
            rand_data = np.random.normal(size=(len(X_batch), model.G_size))
            # update D once
            model.sess.run(model.D_train, {model.G_in: rand_data, model.X_in: X_batch, model.train_flag: True})
            # update G twice
            model.sess.run(model.G_train, {model.G_in: rand_data, model.train_flag: True})
            model.sess.run(model.G_train, {model.G_in: rand_data, model.train_flag: True})
            G_loss, D_loss, G_probs, loss = model.sess.run([model.G_loss, model.D_loss, model.G_true_prob, model.l2_loss],
                                                           {model.G_in: rand_data,
                                                            model.X_in: X_batch,
                                                            model.train_flag: False})
            print ('Epoch %d/%d | Step %d/%d | G loss: %.4f | D loss: %.4f | Proba: %.3f | Loss: %.4f' % 
                   (epoch+1, N_EPOCH, local_step, n_batch, G_loss, D_loss, G_probs.mean(), loss))
        img = model.sess.run(model.G_out_pre, {model.X_in: np.expand_dims(X_test[21], 0),
                                               model.G_in: np.random.normal(size=(1, model.G_size)),
                                               model.train_flag: False})
        plt.imshow(np.squeeze(X_test[21]))
        plt.savefig('./temp/mnist_original')
        plt.imshow(np.squeeze(img))
        plt.savefig('./temp/mnist_gen')
