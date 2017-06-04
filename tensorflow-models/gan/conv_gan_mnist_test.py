from conv_gan_mnist import ConvGAN
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


n_epoch = 5
batch_size = 128


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.mnist.load_data()
    X_train = (X_train / 255.0).reshape(-1, 28, 28, 1)
    X_test = (X_test / 255.0).reshape(-1, 28, 28, 1)
    
    model = ConvGAN((28, 28, 1))
    
    n_batch = len(X_train) / batch_size
    model.sess.run(tf.global_variables_initializer())
    
    for epoch in range(n_epoch):
        X = shuffle(X_train)
        for local_step, X_batch in enumerate(model.gen_batch(X, batch_size)):
            for _ in range(2):
                _, G_loss = model.sess.run([model.G_train, model.G_loss],
                                           {model.G_in: np.random.randn(len(X_batch), model.G_size),
                                            model.train_flag: True})
            _, D_loss, probas = model.sess.run([model.D_train, model.D_loss, model.X_true_prob],
                                               {model.G_in: np.random.randn(len(X_batch), model.G_size),
                                                model.X_in: X_batch,
                                                model.train_flag: True})
            print ('Epoch %d/%d | Step %d/%d | G loss: %.4f | D loss: %.4f | Proba: %.3f' % 
                   (epoch+1, n_epoch, local_step+1, n_batch, G_loss, D_loss, probas.mean()))
        img = model.sess.run(model.G_out, {model.G_in: np.random.randn(1, 100),
                                           model.train_flag: False})
        plt.imshow(np.squeeze(X_test[21]))
        plt.show()
        plt.imshow(np.squeeze(img))
        plt.show()
