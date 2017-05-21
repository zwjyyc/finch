import tensorflow as tf
import math
import matplotlib.pyplot as plt


class RNNRegressor:
    def __init__(self, n_step, n_in, n_out, cell_size=128, sess=tf.Session()):
        """
        Parameters:
        -----------
        n_step: int
            Number of time steps
        n_in: int
            Input dimensions
        n_out: int
            Output dimensions
        cell_size: int
            Number of units in the rnn cell
        sess: object
            tf.Session() object
        """
        self.n_step = n_step
        self.n_in = n_in
        self.n_out = n_out
        self.cell_size = cell_size
        self.sess = sess
        self.current_layer = None
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.add_input_layer()
        self.add_lstm_cells()
        self.add_dynamic_rnn()
        self.add_output_layer() 
        self.add_backward_path()
    # end method build_graph


    def add_input_layer(self):
        self.batch_size = tf.placeholder(tf.int32)
        self.X = tf.placeholder(tf.float32, [None, self.n_step, self.n_in])
        self.Y = tf.placeholder(tf.float32, [None, self.n_step, self.n_out])
        self.W = tf.get_variable('W', [self.cell_size, self.n_out], tf.float32,
                                 tf.contrib.layers.variance_scaling_initializer())
        self.b = tf.get_variable('b', [self.n_out], tf.float32, tf.constant_initializer(0.0))
        self.current_layer = self.X
    # end method add_input_layer


    def add_lstm_cells(self):
        self.cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size)
    # end method add_lstm_cells


    def add_dynamic_rnn(self):
        self.init_state = self.cell.zero_state(self.batch_size, dtype=tf.float32)
        self.current_layer, self.final_state = tf.nn.dynamic_rnn(self.cell, self.current_layer,
                                                                 initial_state=self.init_state,
                                                                 time_major=False)
    # end method add_dynamic_rnn


    def add_output_layer(self):
        reshaped = tf.reshape(self.current_layer, [-1, self.cell_size]) # (batch * n_step, n_hidden)
        # (batch * n_step, n_hidden) dot (n_hidden, n_out)
        self.logits = tf.nn.bias_add(tf.matmul(reshaped, self.W), self.b)
        self.time_seq_out = tf.reshape(self.logits, [-1, self.n_step, self.n_out])
    # end method add_output_layer


    def add_backward_path(self):
        """
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            logits = [tf.reshape(self.logits, [-1])],
            targets = [tf.reshape(self.Y, [-1])],
            weights = [tf.ones([self.batch_size * self.n_step])],
            average_across_timesteps = True,
            softmax_loss_function = self.squared_error,
        )
        self.loss = tf.div(tf.reduce_sum(losses), tf.cast(self.batch_size, tf.float32))
        """
        square_loss = tf.square(tf.subtract(tf.reshape(self.logits, [-1]), tf.reshape(self.Y, [-1])))
        avg_across_steps = tf.div(tf.reduce_sum(square_loss), self.n_step)
        self.loss = tf.div(avg_across_steps, tf.cast(self.batch_size, tf.float32))
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
    # end method add_backward_path
# end class RNNRegressor