import tensorflow as tf
import numpy as np
import math


class BiRNN:
    def __init__(self, seq_len, vocab_size, n_out, embedding_dims=128, cell_size=128, sess=tf.Session()):
        """
        Parameters:
        -----------
        seq_len: int
            Sequence length
        vocab_size: int
            Vocabulary size
        cell_size: int
            Number of units in the rnn cell
        n_out: int
            Output dimensions
        sess: object
            tf.Session() object
        stateful: boolean
            If true, the final state for each batch will be used as the initial state for the next batch 
        """
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.embedding_dims = embedding_dims
        self.cell_size = cell_size
        self.n_out = n_out
        self.sess = sess
        self._cursor = None
        self.build_graph()
    # end constructor


    def build_graph(self):
        with tf.variable_scope('main_model'):
            self.add_input_layer()
            self.add_word_embedding_layer()
            self.add_lstm_cells()
            self.add_dynamic_rnn()
            self.add_output_layer()
            self.add_backward_path()
        with tf.variable_scope('main_model', reuse=True):
            self.add_inference()
    # end method build_graph


    def add_input_layer(self):
        self.X = tf.placeholder(tf.int64, [None, self.seq_len])
        self.Y = tf.placeholder(tf.int64, [None, self.seq_len])
        self.batch_size = tf.placeholder(tf.int32, [])
        self.keep_prob = tf.placeholder(tf.float32)
        self.lr = tf.placeholder(tf.float32)
        self._cursor = self.X
    # end method add_input_layer


    def add_word_embedding_layer(self):
        E = tf.get_variable('E', [self.vocab_size, self.embedding_dims], tf.float32, tf.random_uniform_initializer(-1, 1))
        embedded = tf.nn.embedding_lookup(E, self._cursor)
        self._cursor = tf.nn.dropout(embedded, self.keep_prob)
    # end method add_word_embedding_layer


    def add_lstm_cells(self):
        with tf.variable_scope('forward'):
            self.cell_fw = tf.nn.rnn_cell.LSTMCell(self.cell_size, initializer=tf.orthogonal_initializer)
        with tf.variable_scope('backward'):
            self.cell_bw = tf.nn.rnn_cell.LSTMCell(self.cell_size, initializer=tf.orthogonal_initializer)
    # end method add_rnn_cells


    def add_dynamic_rnn(self):
        with tf.variable_scope('forward'):
            self.fw_out, _ = tf.nn.dynamic_rnn(self.cell_fw, self._cursor, dtype=tf.float32)
        with tf.variable_scope('backward'):
            self.bw_out, _ = tf.nn.dynamic_rnn(self.cell_bw, tf.reverse(self._cursor, [1]), dtype=tf.float32)
        self.bw_out = tf.reverse(self.bw_out, [1])
    # end method add_dynamic_rnn


    def add_output_layer(self):
        fw_logits = tf.layers.dense(tf.reshape(self.fw_out, [-1, self.cell_size]), self.n_out, name='fw_out')
        bw_logits = tf.layers.dense(tf.reshape(self.bw_out, [-1, self.cell_size]), self.n_out, name='bw_out')
        self.logits = fw_logits + bw_logits
    # end method add_output_layer


    def add_backward_path(self):
        self.loss = tf.contrib.seq2seq.sequence_loss(
            logits = tf.reshape(self.logits, [self.batch_size, self.seq_len, self.n_out]),
            targets = self.Y,
            weights = tf.ones([self.batch_size, self.seq_len]),
            average_across_timesteps = True,
            average_across_batch = True,
        )
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1),
                                                   tf.reshape(self.Y, [-1])), tf.float32))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    # end method add_backward_path


    def add_inference(self):
        self.x_fw = tf.placeholder(tf.int32, [None, 1])
        self.x_bw = tf.placeholder(tf.int32, [None, 1])

        self.i_s_fw = self.cell_fw.zero_state(1, tf.float32)
        self.i_s_bw = self.cell_bw.zero_state(1, tf.float32)

        x_fw_embedded = tf.nn.embedding_lookup(tf.get_variable('E'), self.x_fw)
        x_bw_embedded = tf.nn.embedding_lookup(tf.get_variable('E'), self.x_bw)

        with tf.variable_scope('forward', reuse=True):
            y_fw, self.f_s_fw = tf.nn.dynamic_rnn(self.cell_fw, x_fw_embedded, initial_state=self.i_s_fw)
        with tf.variable_scope('backward', reuse=True):
            y_bw, self.f_s_bw = tf.nn.dynamic_rnn(self.cell_bw, x_bw_embedded, initial_state=self.i_s_bw)
        
        self.y_fw = tf.layers.dense(tf.reshape(y_fw, [-1, self.cell_size]), self.n_out, name='fw_out', reuse=True)
        self.y_bw = tf.layers.dense(tf.reshape(y_bw, [-1, self.cell_size]), self.n_out, name='bw_out', reuse=True)
    # end add_sample_model


    def fit(self, X, Y, val_data=None, n_epoch=10, batch_size=128, en_exp_decay=True, en_shuffle=True,
            keep_prob=1.0):
        if val_data is None:
            print("Train %d samples" % len(X) )
        else:
            print("Train %d samples | Test %d samples" % (len(X), len(val_data[0])))
        log = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
        global_step = 0

        self.sess.run(tf.global_variables_initializer()) # initialize all variables
        for epoch in range(n_epoch): # batch training
            if en_shuffle:
                shuffled = np.random.permutation(len(X))
                X = X[shuffled]
                Y = Y[shuffled]
            local_step = 1

            for X_batch, Y_batch in zip(self.gen_batch(X, batch_size),
                                        self.gen_batch(Y, batch_size)):
                lr = self.decrease_lr(en_exp_decay, global_step, n_epoch, len(X), batch_size)           
                _, loss, acc = self.sess.run([self.train_op, self.loss, self.acc],
                                             {self.X:X_batch, self.Y:Y_batch,
                                              self.batch_size:len(X_batch), self.lr:lr,
                                              self.keep_prob:keep_prob})
                local_step += 1
                global_step += 1
                if local_step % 50 == 0:
                    print ('Epoch %d/%d | Step %d/%d | train_loss: %.4f | train_acc: %.4f | lr: %.4f'
                           %(epoch+1, n_epoch, local_step, int(len(X)/batch_size), loss, acc, lr))

            if val_data is not None: # go through testing data, average validation loss and ac 
                val_loss_list, val_acc_list = [], []
                for X_test_batch, Y_test_batch in zip(self.gen_batch(val_data[0], batch_size),
                                                      self.gen_batch(val_data[1], batch_size)):
                    v_loss, v_acc = self.sess.run([self.loss, self.acc],
                                                    {self.X:X_test_batch, self.Y:Y_test_batch,
                                                    self.batch_size:len(X_test_batch),
                                                    self.keep_prob:1.0})
                    val_loss_list.append(v_loss)
                    val_acc_list.append(v_acc)
                val_loss, val_acc = self.list_avg(val_loss_list), self.list_avg(val_acc_list)

            # append to log
            log['loss'].append(loss)
            log['acc'].append(acc)
            if val_data is not None:
                log['val_loss'].append(val_loss)
                log['val_acc'].append(val_acc)

            # verbose
            if val_data is None:
                print ("Epoch %d/%d | train_loss: %.4f | train_acc: %.4f |" % (epoch+1, n_epoch, loss, acc),
                       "lr: %.4f" % (lr) )
            else:
                print ("Epoch %d/%d | train_loss: %.4f | train_acc: %.4f |" % (epoch+1, n_epoch, loss, acc),
                       "test_loss: %.4f | test_acc: %.4f |" % (val_loss, val_acc), "lr: %.4f" % (lr) )
        # end "for epoch in range(n_epoch)"

        return log
    # end method fit


    def predict(self, X_test, batch_size=128):
        batch_pred_list = []
        for X_test_batch in self.gen_batch(X_test, batch_size):
            batch_pred = self.sess.run(self.logits,
                                      {self.X:X_test_batch, self.batch_size:len(X_test_batch),
                                       self.keep_prob:1.0})
            batch_pred_list.append(batch_pred)
        return np.argmax(np.vstack(batch_pred_list), 1)
    # end method predict


    def infer(self, xs):
        n_s_fw = self.sess.run(self.i_s_fw)
        n_s_bw = self.sess.run(self.i_s_bw)
        ys_fw = []
        ys_bw = []
        for x_fw, x_bw in zip(xs, list(reversed(xs))):
            x_fw = np.atleast_2d(x_fw)
            x_bw = np.atleast_2d(x_bw)
            y_fw, y_bw, n_s_fw, n_s_bw = self.sess.run([self.y_fw, self.y_bw, self.f_s_fw, self.f_s_bw], 
                                              {self.x_fw: x_fw,
                                               self.x_bw: x_bw,
                                               self.keep_prob: 1.0,
                                               self.i_s_fw: n_s_fw,
                                               self.i_s_bw: n_s_bw})
            ys_fw.append(y_fw)
            ys_bw.append(y_bw)
        return np.argmax((np.vstack(ys_fw) + np.vstack(ys_bw)[::-1]), 1)
    # end method infer


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method gen_batch


    def decrease_lr(self, en_exp_decay, global_step, n_epoch, len_X, batch_size):
        if en_exp_decay:
            max_lr = 0.005
            min_lr = 0.0005
            decay_rate = math.log(min_lr/max_lr) / (-n_epoch*len_X/batch_size)
            lr = max_lr*math.exp(-decay_rate*global_step)
        else:
            lr = 0.001
        return lr
    # end method adjust_lr


    def list_avg(self, l):
        return sum(l) / len(l)
    # end method list_avg
# end class