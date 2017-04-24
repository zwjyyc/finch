import tensorflow as tf


class MatrixFactorization:
    def __init__(self, n_user, n_item, n_hidden, sess):
        self.n_user = n_user
        self.n_item = n_item
        self.n_hidden = n_hidden
        self.sess = sess
        self.build_graph()

    
    def build_graph(self):
        self.add_input_layer()
        self.add_output_layer()
        self.add_backward_path()


    def add_input_layer(self):
        self.R = tf.placeholder(tf.float32, [self.n_user, self.n_item])
        self.U = tf.Variable(tf.truncated_normal([self.n_user, self.n_hidden]))
        self.I = tf.Variable(tf.truncated_normal([self.n_hidden, self.n_item]))


    def add_output_layer(self):
        self.R_pred = tf.matmul(self.U, self.I)


    def add_backward_path(self):
        R_flatten = tf.reshape(self.R, [-1])
        R_pred_flatten = tf.reshape(self.R_pred, [-1])
        zeros = tf.zeros([self.n_user*self.n_item])
        self.non_zero_indices = tf.where(tf.not_equal(R_flatten, zeros))
        R_non_zero = tf.gather(R_flatten, self.non_zero_indices)
        R_pred_non_zero = tf.gather(R_pred_flatten, self.non_zero_indices)
        cost = tf.reduce_sum(tf.abs(tf.subtract(R_non_zero, R_pred_non_zero)))

        lda = tf.constant(.001)
        norm_sums = tf.add(tf.reduce_sum(tf.abs(self.U)),  tf.reduce_sum(tf.abs(self.I)))
        regu = tf.multiply(norm_sums, lda)
        
        self.loss = tf.add(cost, regu)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
# end class