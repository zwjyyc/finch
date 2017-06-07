from mlp_gan import MLP_GAN
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


batch_size = 64
n_X_in = 15
n_G_in = 5
x_range = np.vstack([np.linspace(-1, 1, n_X_in) for _ in range(batch_size)])


def load_data():
    a = np.random.uniform(1, 2, size=batch_size)[:, np.newaxis]
    data = a * np.power(x_range, 2) + (a-1)
    return data


if __name__ == '__main__':
    model = MLP_GAN(n_G_in, n_X_in)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    plt.ion()
    plt.show()
    
    for step in range(3000):
        rand_data = np.random.randn(batch_size, n_G_in)
        _, G_out = sess.run([model.G_train, model.G_out], {model.G_in: rand_data})
        sess.run(model.D_train, {model.G_in: rand_data, model.X_in: load_data()})
        G_loss, D_loss, D_prob, G_prob, loss = sess.run([model.G_loss, model.D_loss, model.X_true_prob, model.G_true_prob,
                                                         model.l2_loss],
                                                        {model.G_in: rand_data, model.X_in: load_data()})
        if step % 50 == 0:
            plt.cla()
            plt.plot(x_range[0], G_out[0], c='#4AD631', lw=3, label='generated',)
            plt.plot(x_range[0], 2 * np.power(x_range[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
            plt.plot(x_range[0], 1 * np.power(x_range[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
            plt.legend(loc='upper right', fontsize=12)
            plt.draw()
            plt.pause(0.01)
            print(step)
        print("G loss: %.4f | D loss: %.4f | D prob: %.4f | G prob: %.4f | l2 loss: %.4f " %
              (G_loss, D_loss, D_prob.mean(), G_prob.mean(), loss))
    plt.ioff()
    plt.show()
