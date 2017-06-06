from conv_gan_cifar10 import ConvGAN
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


N_EPOCH = 10
BATCH_SIZE = 128


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.cifar10.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    model = ConvGAN((32, 32, 3))
    
    n_batch = len(X_train) / BATCH_SIZE
    model.sess.run(tf.global_variables_initializer())
    
    for epoch in range(N_EPOCH):
        X = shuffle(X_train)
        for local_step, X_batch in enumerate(model.gen_batch(X, BATCH_SIZE)):
            _, G_loss = model.sess.run([model.G_train, model.G_loss],
                                       {model.G_in: np.random.randn(len(X_batch), model.G_size),
                                        model.train_flag: True})
            _, D_loss, probas, mse = model.sess.run([model.D_train, model.D_loss, model.X_true_prob, model.mse_loss],
                                                    {model.G_in: np.random.randn(len(X_batch), model.G_size),
                                                     model.X_in: X_batch,
                                                     model.train_flag: True})
            print ('Epoch %d/%d | Step %d/%d | G loss: %.4f | D loss: %.4f | Proba: %.3f | MSE: %.4f' % 
                   (epoch+1, N_EPOCH, local_step, n_batch, G_loss, D_loss, probas.mean(), mse))
        img = model.sess.run(model.G_out_pre, {model.G_in: np.random.randn(1, 100),
                                               model.train_flag: False})
        plt.imshow(X_test[21])
        plt.savefig('./temp/cifar10_original')
        plt.imshow(np.squeeze(img))
        plt.savefig('./temp/cifar10_gen')
