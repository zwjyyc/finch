from mlp_cond_gan import MLP_GAN
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


batch_size = 64
X_size = 15
G_size = 5
x_range = np.vstack([np.linspace(-1, 1, X_size) for _ in range(batch_size)])


def load_data():
    a = np.random.uniform(1, 2, size=batch_size)[:, np.newaxis]
    data = a * np.power(x_range, 2) + (a-1)
    labels = (a - 1) > 0.5  # upper paintings (1), lower paintings (0), two classes
    labels = labels.astype(np.float32)
    return data, labels


if __name__ == '__main__':
    model = MLP_GAN(G_size, X_size)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    plt.ion()
    plt.show()
    
    for step in range(5000):
        rand_data = np.random.randn(batch_size, G_size)
        real_data, labels = load_data()

        _, G_out = sess.run([model.G_train, model.G_out], {model.G_in: rand_data,
                                                           model.labels: labels})

        _ = sess.run(model.D_train, {model.G_in: rand_data, model.X_in: real_data,
                                                            model.labels: labels})
        
        G_loss, D_loss, D_prob, G_prob, loss = sess.run([model.G_loss, model.D_loss,
                                                         model.X_true_prob, model.G_true_prob,
                                                         model.l2_loss],
                                                        {model.G_in: rand_data,
                                                         model.X_in: real_data,
                                                         model.labels: labels})
        if step % 50 == 0:
            plt.cla()
            plt.plot(x_range[0], G_out[0], c='#4AD631', lw=3, label='generated')
            bound = [0, 0.5] if labels[0, 0] == 0 else [0.5, 1]
            plt.plot(x_range[0], 2 * np.power(x_range[0], 2) + bound[1], c='#74BCFF', lw=3, label='upper bound')
            plt.plot(x_range[0], 1 * np.power(x_range[0], 2) + bound[0], c='#FF9359', lw=3, label='lower bound')
            plt.text(-0.5, 1.7, 'Class = %i' % int(labels[0, 0]), fontdict={'size': 15})
            plt.ylim((0, 3))
            plt.legend(loc='upper right', fontsize=12)
            plt.draw()
            plt.pause(0.1)
        if step % 200 == 0:
            print("Step %d |  G loss: %.4f | D loss: %.4f | D prob: %.4f | G prob: %.4f | l2 loss: %.4f " %
                 (step, G_loss, D_loss, D_prob.mean(), G_prob.mean(), loss))
    plt.ioff()

    plt.figure(2) # plot a generated painting for upper class
    z = np.random.randn(1, G_size)
    label = np.array([[1.0]]) # for upper class
    G_paintings = sess.run(model.G_out, {model.G_in: z, model.labels: label})
    plt.plot(x_range[0], G_paintings[0], c='#4AD631', lw=3, label='G painting for upper class',)
    plt.plot(x_range[0], 2 * np.power(x_range[0], 2) + bound[1], c='#74BCFF', lw=3, label='upper bound (class 1)')
    plt.plot(x_range[0], 1 * np.power(x_range[0], 2) + bound[0], c='#FF9359', lw=3, label='lower bound (class 0)')
    plt.ylim((0, 3))
    plt.legend(loc='upper right', fontsize=12)
    plt.show()
