import tensorflow as tf
import math
import matplotlib.pyplot as plt


class RNNRegressor:
    def __init__(self, n_in, n_step, n_hidden, n_out):
        self.n_in = n_in
        self.n_step = n_step
        self.n_hidden = n_hidden
        self.n_out = n_out

        self.build_graph()
    # end constructor

    def build_graph(self):
        self.batch_size = tf.placeholder(tf.int32)
        self.X = tf.placeholder(tf.float32, [None, self.n_step, self.n_in])
        self.y = tf.placeholder(tf.float32, [None, self.n_step, self.n_out])
        W = tf.Variable(tf.truncated_normal([self.n_hidden, self.n_out], stddev=math.sqrt(2/self.n_hidden)))
        b = tf.Variable(tf.zeros([self.n_out]))
        self.pred = self.rnn(self.X, W, b)
        self.loss = tf.reduce_mean(tf.square(self.pred - self.y))
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
    # end method build_graph


    def rnn(self, X, W, b):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)
        self.init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        outputs, self.final_state = tf.nn.dynamic_rnn(lstm_cell, X, initial_state=self.init_state,
                                                      time_major=False)
        
        outputs = tf.reshape(outputs, [-1, self.n_hidden]) # (batch * n_step, n_hidden)
        outputs = tf.matmul(outputs, W) + b # (batch * n_step, n_hidden) dot (n_hidden, n_out)
        return tf.reshape(outputs, [-1, self.n_step, self.n_out])
    # end method rnn


    def fit(self, train_data, batch_size, test_data=None):
        self.sess.run(self.init)
        train_state = self.sess.run(self.init_state, feed_dict={self.batch_size:batch_size})
        for train_idx, train_sample in enumerate(train_data):
            seq, res = train_sample
            _, train_loss, train_state = self.sess.run( [self.train_op, self.loss, self.final_state],
                feed_dict={self.X: seq, self.y: res, self.init_state: train_state, self.batch_size: batch_size})
            if test_data is None:
                if train_idx % 20 == 0:
                    print('train loss: %.4f' % (train_loss))
            else:
                test_loss_list = []
                test_state = self.sess.run(self.init_state, feed_dict={self.batch_size:batch_size})
                for test_idx, test_sample in enumerate(test_data):
                    seq_test, res_test = test_sample
                    test_loss, test_state = self.sess.run([self.loss, self.final_state],
                        feed_dict={self.X: seq_test, self.y: res_test, self.init_state: test_state,
                                   self.batch_size: batch_size})
                    test_loss_list.append(test_loss)
                if train_idx % 20 == 0:
                    print('train loss: %.4f |' % (train_loss),
                          'test loss: %.4f' % (sum(test_loss_list)/len(test_loss_list)) )
    # end method fit


    def fit_plot(self, train_data, batch_size, test_data):
        self.sess.run(self.init)
        plt.ion()
        plt.show()

        train_state = self.sess.run(self.init_state, feed_dict={self.batch_size:batch_size})
        test_state = self.sess.run(self.init_state, feed_dict={self.batch_size:batch_size})
        for train_idx, train_sample in enumerate(train_data):
            seq, res, xs = train_sample
            _, train_loss, train_state = self.sess.run( [self.train_op, self.loss, self.final_state],
                feed_dict={self.X: seq, self.y: res, self.init_state: train_state, self.batch_size: batch_size})

            test_sample = test_data[train_idx]
            seq_test, res_test, xs_test = test_sample
            test_loss, test_state, test_pred = self.sess.run([self.loss, self.final_state, self.pred],
                feed_dict={self.X: seq_test, self.y: res_test, self.init_state: test_state,
                    self.batch_size: batch_size})
            
            # update plotting
            plt.plot(xs.ravel(), res_test.ravel(), 'r', xs.ravel(), test_pred.ravel(), 'b--')
            plt.ylim((-1.2, 1.2))
            plt.xlim((xs.ravel()[0], xs.ravel()[-1]))
            plt.draw()
            plt.pause(0.3)

            if train_idx % 20 == 0:
                print('train loss: %.4f | test loss: %.4f' % (train_loss, test_loss))         
    # end method fit
# end class RNNRegressor
