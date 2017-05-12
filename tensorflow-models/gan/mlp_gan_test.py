from mlp_gan import MLP_GAN
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


batch_size = 64
n_D_in = 15
n_G_in = 5
x_range = np.vstack([np.linspace(-1, 1, n_D_in) for _ in range(batch_size)])


def load_data():
    a = np.random.uniform(1, 2, size=batch_size)[:, np.newaxis]
    data = a * np.power(x_range, 2) + (a-1)
    return data


if __name__ == '__main__':
    GAN = MLP_GAN(n_D_in, n_G_in)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    plt.ion()
    plt.show()
    
    for step in range(5000):
        G_out, D_prob, D_loss, _, _ = sess.run([GAN.G_out, GAN.D_prob, GAN.D_loss, GAN.D_train, GAN.G_train],
                                        {GAN.G_in: np.random.randn(batch_size, n_G_in),
                                         GAN.D_in: load_data()})
        if step % 50 == 0:
            plt.cla()
            plt.plot(x_range[0], G_out[0], c='#4AD631', lw=3, label='generated',)
            plt.plot(x_range[0], 2 * np.power(x_range[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
            plt.plot(x_range[0], 1 * np.power(x_range[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
            plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)'%D_prob.mean(), fontdict={'size': 15})
            plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)'%-D_loss, fontdict={'size': 15})
            plt.ylim((0, 3))
            plt.legend(loc='upper right', fontsize=12)
            plt.draw()
            plt.pause(0.01)
            print(step)
    plt.ioff()
    plt.show()
