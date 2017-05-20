from rnn_regr import RNNRegressor
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class TimeSeriesGen:
    def __init__(self, batch_start=0, time_steps=20, batch_size=50):
        self.start = batch_start
        self.t_steps = time_steps
        self.size = batch_size
        
    def next_batch(self):
        xs = np.arange(self.start, self.start + self.t_steps * self.size)
        xs = xs.reshape([self.size, self.t_steps]) / (10*np.pi)
        self.start += self.t_steps * self.size
        seq = np.sin(xs)
        res = np.cos(xs)
        return (seq[:, :, np.newaxis], res[:, :, np.newaxis], xs)


if __name__ == '__main__':
    train_gen = TimeSeriesGen()
    test_gen = TimeSeriesGen(1000)
    model = RNNRegressor(n_step = 20,
                         n_in = 1,
                         n_out = 1,
                         cell_size = 16)

    model.sess.run(tf.global_variables_initializer())
    train_state = model.sess.run(model.init_state, feed_dict={model.batch_size:50})
    test_state = model.sess.run(model.init_state, feed_dict={model.batch_size:50})

    for _ in range(1000):
        seq_train, res_train, _ = train_gen.next_batch()
        _, train_loss, train_state = model.sess.run([model.train_op, model.loss, model.final_state],
                                                     feed_dict={model.X:seq_train, model.Y:res_train,
                                                                model.init_state:train_state,
                                                                model.batch_size:50})

        seq_test, res_test, xs = test_gen.next_batch()
        test_loss, test_state, test_pred = model.sess.run([model.loss, model.final_state, model.time_seq_out],
                                                          {model.X:seq_test, model.Y:res_test,
                                                           model.init_state:test_state,
                                                           model.batch_size:50})

        
        # update plotting
        plt.plot(xs.ravel(), res_test.ravel(), 'r', xs.ravel(), test_pred.ravel(), 'b--')
        plt.ylim((-1.2, 1.2))
        plt.xlim((xs.ravel()[0], xs.ravel()[-1]))
        plt.draw()
        plt.pause(0.3)

        print('train loss: %.4f | test loss: %.4f' % (train_loss, test_loss)) 
