from rnn_regr import RNNRegressor
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


TIME_STEPS = 20
BATCH_SIZE = 50


class TimeSeriesGen:
    def __init__(self, batch_start, time_steps, batch_size):
        self.batch_start = batch_start
        self.time_steps = time_steps
        self.batch_size = batch_size
        
    def next_batch(self):
        xs = np.arange(self.batch_start, self.batch_start + self.time_steps * self.batch_size)
        xs = xs.reshape([self.batch_size, self.time_steps]) / (10 * np.pi)
        self.batch_start += self.time_steps * self.batch_size
        seq = np.sin(xs)
        res = np.cos(xs)
        return (seq[:, :, np.newaxis], res[:, :, np.newaxis], xs)


if __name__ == '__main__':
    train_gen = TimeSeriesGen(0, TIME_STEPS, BATCH_SIZE)
    test_gen = TimeSeriesGen(1000, TIME_STEPS, BATCH_SIZE)
    model = RNNRegressor(n_step = 20,
                         n_in = 1,
                         n_out = 1,
                         cell_size = 16)

    model.sess.run(tf.global_variables_initializer())
    train_state = model.sess.run(model.init_state, feed_dict={model.batch_size:BATCH_SIZE})
    test_state = model.sess.run(model.init_state, feed_dict={model.batch_size:BATCH_SIZE})

    for _ in range(1000):
        seq_train, res_train, _ = train_gen.next_batch()
        _, train_loss, train_state = model.sess.run([model.train_op, model.loss, model.final_state],
                                                     feed_dict={model.X:seq_train,
                                                                model.Y:res_train,
                                                                model.init_state:train_state,
                                                                model.batch_size:BATCH_SIZE})

        seq_test, res_test, xs = test_gen.next_batch()
        test_loss, test_state, test_pred = model.sess.run([model.loss, model.final_state, model.time_seq_out],
                                                          {model.X:seq_test,
                                                           model.Y:res_test,
                                                           model.init_state:test_state,
                                                           model.batch_size:BATCH_SIZE})

        # update plotting
        plt.plot(xs.ravel(), res_test.ravel(), 'r', xs.ravel(), test_pred.ravel(), 'b--')
        plt.ylim((-1.2, 1.2))
        plt.xlim((xs.ravel()[0], xs.ravel()[-1]))
        plt.draw()
        plt.pause(0.3)

        print('train loss: %.4f | test loss: %.4f' % (train_loss, test_loss)) 
