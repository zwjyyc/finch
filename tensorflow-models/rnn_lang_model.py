import tensorflow as tf
import math


class RNNLangModel:
    def __init__(self, n_hidden, n_layers, vocab_size, seq_len):
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.batch_size = tf.placeholder(tf.int32)
        self.X = tf.placeholder(tf.int32, [None, self.seq_len])
        self.Y = tf.placeholder(tf.int32, [None, self.seq_len])
        W = tf.Variable(tf.random_normal([self.n_hidden, self.vocab_size], stddev=math.sqrt(2/self.n_hidden)))
        b = tf.Variable(tf.zeros([self.vocab_size]))
        """
        X from (batch_size, seq_len) -> (batch_size, seq_len, n_hidden)
        where each word in (batch_size, seq_len) is represented by a vector of length [n_hidden]
        """
        embedding_mat = tf.Variable(tf.random_normal([self.vocab_size, self.n_hidden]))
        embedding_output = tf.nn.embedding_lookup(embedding_mat, self.X)
        self.pred = tf.matmul(self.rnn(embedding_output), W) + b
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred,
                                                                     labels=tf.reshape(self.Y, [-1]))
        self.loss = tf.reduce_mean(losses)
        self.lr = tf.placeholder(tf.float32)
        """
        gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tf.trainable_variables()), 4.5)
        optimizer = tf.train.AdamOptimizer()
        self.train_op = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))
        """
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
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


    def adjust_lr(self, en_exp_decay, global_step, n_epoch, nb_batch):
        if en_exp_decay:
            max_lr = 0.003
            min_lr = 0.0001
            decay_rate = math.log(min_lr/max_lr) / (-n_epoch*nb_batch)
            lr = max_lr*math.exp(-decay_rate*global_step)
        else:
            lr = 0.001
        return lr
    # end method get_lr


    def fit(self, X_batch_list, Y_batch_list, n_epoch=10, batch_size=128, en_exp_decay=True):
        log = {'train_loss': []}
        global_step = 0
        self.sess.run(self.init) # initialize all variables
        for epoch in range(n_epoch):
            next_state = self.sess.run(self.init_state, feed_dict={self.batch_size:batch_size})
            batch_count = 1
            for X_batch, Y_batch in zip(X_batch_list, Y_batch_list):
                lr = self.adjust_lr(en_exp_decay, global_step, n_epoch, len(X_batch_list))
                _, loss, next_state = self.sess.run([self.train_op, self.loss, self.final_state],
                    feed_dict={self.X:X_batch, self.Y:Y_batch, self.init_state:next_state,
                               self.batch_size:batch_size, self.lr:lr})
                if batch_count % 10 == 0:
                    print ('Epoch %d/%d | Batch %d/%d | train loss: %.4f | lr: %.4f' % (epoch+1, n_epoch,
                        batch_count, len(X_batch_list), loss, lr))
                log['train_loss'].append(loss)
                batch_count += 1
                global_step += 1
        return log
    # end method fit
# end class CharRNNModel
