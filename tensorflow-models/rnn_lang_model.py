import tensorflow as tf
import random
from utils import to_one_hot
import numpy as np


class RNNLangModel:
    def __init__(self, n_hidden, n_layers, vocab_size, seq_len):
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.X = tf.placeholder(tf.int32, [None, self.seq_len])
        self.Y = tf.placeholder(tf.int32, [None, self.seq_len])
        self.W = tf.random_normal([self.n_hidden, self.vocab_size])
        self.b = tf.random_normal([self.vocab_size])
        self.batch_size = tf.placeholder(tf.int32)

        """
        X from (batch_size, seq_len) -> (batch_size, seq_len, n_hidden)
        where each word is represented by a vector of length [n_hidden]
        """
        embedding_mat = tf.random_normal([self.vocab_size, self.n_hidden])
        embedding_output = tf.nn.embedding_lookup(embedding_mat, self.X)

        self.pred = tf.matmul(self.rnn(embedding_output), self.W) + self.b
        self.loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                logits = [self.pred],
                targets = [tf.reshape(self.Y, [-1])],
                weights = [tf.ones([self.batch_size * self.seq_len])],
                average_across_timesteps = self.vocab_size,
        )
        self.cost = tf.reduce_sum(self.loss) / tf.cast((self.batch_size*self.seq_len), tf.float32)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
    # end method build_graph

    
    def rnn(self, X):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)
        lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * self.n_layers)
        self.init_state = lstm_cell.zero_state(self.batch_size, tf.float32)
        output, self.final_state = tf.nn.dynamic_rnn(lstm_cell, X, initial_state=self.init_state,
                                                     time_major=False)
        output = tf.reshape(output, [-1, self.n_hidden])
        return output 
    # end method rnn


    def fit(self, X_list, Y_list, n_epoch=5, batch_size=100):
        self.sess.run(self.init) # initialize all variables
        next_state = self.sess.run(self.init_state, feed_dict={self.batch_size:batch_size})
        for epoch in range(n_epoch):
            local_step = 0
            for X_batch, Y_batch in zip(X_list, Y_list):
                _, loss, next_state = self.sess.run([self.train_op, self.cost, self.final_state],
                    feed_dict={self.X:X_batch, self.Y:Y_batch, self.init_state:next_state,
                        self.batch_size:batch_size})
                print ('Epoch %d/%d | Step %d/%d | train loss: %.4f' % (epoch+1, n_epoch, local_step+1,
                        len(X_list), loss))
                local_step += 1
    # end method fit
# end class CharRNNModel
