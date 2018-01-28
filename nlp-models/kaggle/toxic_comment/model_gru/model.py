from config import args

import tensorflow as tf


class Model:
    def __init__(self, params):
        self.params = params
        self.build_graph()
        self.submit= 'result.csv'

    def build_graph(self):
        self.interfaces()
        self.forward()
        self.backward()

    def interfaces(self):
        self.placeholders = {
            'X': tf.placeholder(tf.int32, [None, args.max_len]),
            'Y': tf.placeholder(tf.float32, [None, self.params['n_class']]),
            'train_flag': tf.placeholder(tf.bool)}
        self.ops = {
            'logits': None,
            'predict': None,
            'loss': None,
            'train': None}
    
    def forward(self):
        non_zero = tf.count_nonzero(self.placeholders['X'], 1)
        batch_size = tf.shape(self.placeholders['X'])[0]

        embedding = tf.Variable(self.params['embedding'], dtype=tf.float32,
            trainable=False)
        x = tf.nn.embedding_lookup(embedding, self.placeholders['X'])
        
        (fw_out, bw_out), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
            self.rnn_cell(args.rnn_size//2), self.rnn_cell(args.rnn_size//2),
            x, non_zero, dtype=tf.float32)
        x = tf.concat([fw_out, bw_out], -1)
        
        x = tf.layers.max_pooling1d(x, x.get_shape().as_list()[1], 1)
        x = tf.reshape(x, [batch_size, args.rnn_size])

        x = tf.layers.dropout(x, 0.2, training=self.placeholders['train_flag'])
        x = tf.layers.dense(x, 50, tf.nn.elu)
        
        self.ops['logits'] = tf.layers.dense(x, self.params['n_class'])
        self.ops['predict'] = tf.nn.sigmoid(self.ops['logits'])
    
    def backward(self):
        global_step = tf.Variable(0, trainable=False)
        
        self.ops['lr'] = tf.train.exponential_decay(1e-3, global_step, 10000, 0.1)

        self.ops['loss'] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.placeholders['Y'], logits=self.ops['logits']))

        params = tf.trainable_variables()
        grads = tf.gradients(self.ops['loss'], params)
        clipped_grads, _ = tf.clip_by_global_norm(grads, args.clip_norm)
        self.ops['train'] = tf.train.AdamOptimizer(self.ops['lr']).apply_gradients(
            zip(clipped_grads, params), global_step)

    def train_batch(self, sess, x, y):
        loss, lr, _ = sess.run([self.ops['loss'], self.ops['lr'], self.ops['train']],
            {self.placeholders['X']: x,
             self.placeholders['Y']: y,
             self.placeholders['train_flag']: True})
        return loss, lr

    def test_batch(self, sess, x, y):
        loss = sess.run(self.ops['loss'],
            {self.placeholders['X']: x,
             self.placeholders['Y']: y,
             self.placeholders['train_flag']: False})
        return loss

    def predict_batch(self, sess, x):
        prediction = sess.run(self.ops['predict'],
            {self.placeholders['X']: x,
             self.placeholders['train_flag']: False})
        return prediction

    def rnn_cell(self, rnn_size=None):
        rnn_size = args.rnn_size if rnn_size is None else rnn_size

        keep_prob = tf.cond(self.placeholders['train_flag'],
                            lambda: tf.constant(0.8),
                            lambda: tf.constant(1.0))

        return tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(
            rnn_size, kernel_initializer=tf.orthogonal_initializer()),
                keep_prob)
