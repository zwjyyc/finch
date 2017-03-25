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
        self.X = tf.placeholder(tf.float32, [None, self.n_step, self.n_in])
        self.y = tf.placeholder(tf.float32, [None, self.n_step, self.n_out])
        self.W = {
            'in': tf.Variable(tf.random_normal([self.n_in, self.n_hidden],
                                               stddev=math.sqrt(2.0/self.n_in))),
            'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_out],
                                                stddev=math.sqrt(2.0/self.n_hidden)))
        }
        self.b = {
            'in': tf.Variable(tf.zeros([self.n_hidden])),
            'out': tf.Variable(tf.zeros([self.n_out]))
        }
        self.batch_size = tf.placeholder(tf.int32)
        self.pred = self.rnn(self.X, self.W, self.b)
        self.loss = tf.reduce_mean(tf.square(self.pred - self.y))
        """
        self.losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            logits = [tf.reshape(self.pred, [-1])],
            targets = [tf.reshape(self.y, [-1])],
            weights = [tf.ones([self.batch_size * self.n_step], dtype=tf.float32)],
            average_across_timesteps = True,
            softmax_loss_function = self.ms_error,
            name = 'losses'
        )
        self.loss = tf.reduce_sum(self.losses, name='losses_sum') / tf.cast(self.batch_size, tf.float32)
        """
        self.train = tf.train.AdamOptimizer().minimize(self.loss)
        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
    # end method build_graph


    def rnn(self, X, W, b):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)
        self.init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        outputs, self.final_state = tf.nn.dynamic_rnn(lstm_cell, X, initial_state=self.init_state,
                                                      time_major=False)
        
        outputs = tf.reshape(outputs, [-1, self.n_hidden]) # (batch * n_step, n_hidden)
        outputs = tf.matmul(outputs, self.W['out']) + self.b['out'] # (batch * n_step, n_hidden) dot (n_hidden, n_out)
        return tf.reshape(outputs, [-1, self.n_step, self.n_out])
    # end method rnn


    def fit(self, train_data, batch_size, test_data=None):
        self.sess.run(self.init)
        for train_idx, train_sample in enumerate(train_data):
            seq, res = train_sample
            if train_idx == 0:
                feed_dict_train = {self.X: seq, self.y: res, self.batch_size: batch_size}
            else:
                feed_dict_train = {self.X: seq, self.y: res, self.init_state: train_state,
                                   self.batch_size: batch_size}
            _, train_loss, train_state = self.sess.run(
                [self.train, self.loss, self.final_state], feed_dict=feed_dict_train)

            if test_data is None:
                if train_idx % 20 == 0:
                    print('train loss: %.4f' % (train_loss))
            else:
                test_loss_list = []
                for test_idx, test_sample in enumerate(test_data):
                    seq_test, res_test = test_sample
                    if test_idx == 0:
                        feed_dict_test = {self.X: seq_test, self.y: res_test, self.batch_size: batch_size}
                    else:
                        feed_dict_test = {self.X: seq_test, self.y: res_test, self.init_state: test_state,
                                          self.batch_size: batch_size}
                    test_loss, test_state = self.sess.run([self.loss, self.final_state], feed_dict=feed_dict_test)
                    test_loss_list.append(test_loss)
                
                if train_idx % 20 == 0:
                    print('train loss: %.4f |' % (train_loss),
                          'test loss: %.4f' % (sum(test_loss_list)/len(test_loss_list)) )
    # end method fit


    def fit_plot(self, train_data, batch_size, test_data):
        self.sess.run(self.init)
        plt.ion()
        plt.show()

        for train_idx, train_sample in enumerate(train_data):
            seq, res, xs = train_sample
            if train_idx == 0:
                feed_dict_train = {self.X: seq, self.y: res, self.batch_size: batch_size}
            else:
                feed_dict_train = {self.X: seq, self.y: res, self.init_state: train_state,
                                   self.batch_size: batch_size}
            _, train_loss, train_state = self.sess.run(
                [self.train, self.loss, self.final_state], feed_dict=feed_dict_train)

            test_sample = test_data[train_idx]
            seq_test, res_test, xs_test = test_sample
            if train_idx == 0:
                feed_dict_test = {self.X: seq_test, self.y: res_test, self.batch_size: batch_size}
            else:
                feed_dict_test = {self.X: seq_test, self.y: res_test, self.init_state: test_state,
                                  self.batch_size: batch_size}
            test_loss, test_state, test_pred = self.sess.run([self.loss, self.final_state, self.pred],
                                                              feed_dict=feed_dict_test)
            
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
