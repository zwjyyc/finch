import tensorflow as tf


class RNNRegressor:
    def __init__(self, n_in, n_out, cell_size, sess=tf.Session()):
        self.n_in = n_in
        self.n_out = n_out
        self.cell_size = cell_size
        self.sess = sess
        self._pointer = None
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
        self.X = tf.placeholder(tf.float32, [None, None, self.n_in])
        self.Y = tf.placeholder(tf.float32, [None, None, self.n_out])
        self.batch_size = tf.placeholder(tf.int32, [])
        self._pointer = self.X
    # end method add_input_layer


    def add_lstm_cells(self):
        self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size)
    # end method add_lstm_cells


    def add_dynamic_rnn(self):
        self.init_state = self.cell.zero_state(self.batch_size, dtype=tf.float32)
        self._pointer, self.final_state = tf.nn.dynamic_rnn(
            self.cell, self._pointer, initial_state=self.init_state, time_major=False)
    # end method add_dynamic_rnn


    def add_output_layer(self):
        reshaped = tf.reshape(self._pointer, [-1, self.cell_size])
        self.logits = tf.layers.dense(reshaped, self.n_out)
        self.time_seq_out = tf.reshape(self.logits, [self.batch_size, -1, self.n_out])
    # end method add_output_layer


    def add_backward_path(self):
        flatten = lambda tensor: tf.reshape(tensor, [-1])
        self.loss = tf.reduce_sum(tf.squared_difference(flatten(self.logits), flatten(self.Y))) \
            / tf.cast(tf.shape(self.X)[0], tf.float32) \
            / tf.cast(tf.shape(self.X)[1], tf.float32)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
    # end method add_backward_path
# end class RNNRegressor