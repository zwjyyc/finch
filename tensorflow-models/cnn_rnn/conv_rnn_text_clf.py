import tensorflow as tf
import numpy as np
import math
import sklearn


class ConvLSTMClassifier:
    def __init__(self, seq_len, vocab_size, n_out, sess=tf.Session(),
                 embedding_dims=128, n_filters=64, kernel_size=5, pool_size=4, padding='VALID',
                 cell_size=128):
        """
        Parameters:
        -----------
        seq_len: int
            Sequence length
        vocab_size: int
            Vocabulary size
        embedding_dims: int
            Word embedding dimensions
        n_filters: int
            Number output of filters in the convolution
        kernel_size: int
            Size of the 1D convolution window
        pool_size: int
            Size of the max pooling windows
        cell_size: int
            Number of units in the rnn cell
        n_out: int
            Output dimensions
        sess: object
            tf.Session() object 
        """
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.embedding_dims = embedding_dims
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.padding = padding
        self.cell_size = cell_size
        self.n_out = n_out
        self.sess = sess
        self.current_layer = None               # used to point to the forefront of neural network
        self.current_seq_len = self.seq_len     # used to record the current sequence length (after pooling)
        self.build_graph()
    # end constructor
 
 
    def build_graph(self):
        self.add_input_layer()
        self.add_word_embedding()
        self.add_conv1d('conv', filter_shape=[self.kernel_size, self.embedding_dims, self.n_filters])
        self.add_maxpool(self.pool_size)
        self.add_highway('highway')
        self.add_lstm_cells()
        self.add_dynamic_rnn()
        self.add_output_layer()   
        self.add_backward_path()
    # end method build_graph
 
 
    def add_input_layer(self):
        self.X = tf.placeholder(tf.int32, [None, self.seq_len])
        self.Y = tf.placeholder(tf.float32, [None, self.n_out])
        self.batch_size = tf.placeholder(tf.int32)
        self.keep_prob = tf.placeholder(tf.float32)
        self.lr = tf.placeholder(tf.float32)
        self.train_flag = tf.placeholder(tf.bool)
        self.current_layer = self.X
    # end method add_input_layer


    def add_word_embedding(self):
        E = tf.get_variable('E', [self.vocab_size,self.embedding_dims], tf.float32, tf.random_normal_initializer())
        E = tf.nn.embedding_lookup(E, self.current_layer)
        self.current_layer = tf.nn.dropout(E, self.keep_prob)
    # end method add_word_embedding


    def add_conv1d(self, name, filter_shape, stride=1):
        W = self._W(name+'_w', filter_shape)
        b = self._b(name+'_b', [filter_shape[-1]])                                
        conv = tf.nn.conv1d(self.current_layer, W, stride=stride, padding=self.padding)
        conv = tf.nn.bias_add(conv, b)
        conv = tf.contrib.layers.batch_norm(conv, is_training=self.train_flag)
        conv = tf.nn.relu(conv)
        self.current_layer = conv
        if self.padding == 'VALID':
            self.current_seq_len = int((self.current_seq_len-self.kernel_size+1) / stride)
        if self.padding == 'SAME':
            self.current_seq_len = int(self.current_seq_len / stride)
    # end method add_conv1d


    def add_maxpool(self, k=2):
        conv = tf.expand_dims(self.current_layer, 1)
        conv = tf.nn.max_pool(conv, ksize=[1,1,k,1], strides=[1,1,k,1], padding=self.padding)
        conv = tf.squeeze(conv)
        self.current_seq_len = int(self.current_seq_len / k)
        self.current_layer = tf.reshape(conv, [self.batch_size, self.current_seq_len * self.n_filters])
    # end method add_maxpool


    def add_highway(self, name, carry_bias=-1.0):
        X = self.current_layer
        size = self.current_seq_len * self.n_filters

        W_T = self._W(name+'_wt', [size, size])
        b_T = tf.get_variable(name+'_bt', [size], tf.float32, tf.constant_initializer(carry_bias))
        W = self._W(name+'_w', [size,size])
        b = self._b(name+'_b', [size])

        T = tf.sigmoid(tf.nn.bias_add(tf.matmul(X, W_T), b_T))
        H = tf.nn.relu(tf.nn.bias_add(tf.matmul(X, W), b))
        C = tf.subtract(1.0, T, name="carry_gate")

        Y = tf.add(tf.multiply(H, T), tf.multiply(X, C))
        self.current_layer = tf.reshape(Y, [self.batch_size, self.current_seq_len, self.n_filters])
    # end method add_highway


    def add_lstm_cells(self):
        self.cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size)
    # end method add_rnn_cells


    def add_dynamic_rnn(self):
        self.init_state = self.cell.zero_state(self.batch_size, tf.float32)              
        self.current_layer, final_state = tf.nn.dynamic_rnn(self.cell, self.current_layer,
                                                            initial_state=self.init_state,
                                                            time_major=False)
    # end method add_dynamic_rnn


    def add_output_layer(self):
        # (batch, n_step, n_hidden) -> (n_step, batch, n_hidden) -> n_step * [(batch, n_hidden)]
        time_major = tf.unstack(tf.transpose(self.current_layer, [1,0,2]))
        W = tf.get_variable('softmax_w', [self.cell_size, self.n_out], tf.float32,
                             tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable('softmax_b', [self.n_out], tf.float32, tf.constant_initializer(0.0))
        self.logits = tf.nn.bias_add(tf.matmul(time_major[-1], W), b)
    # end method add_output_layer


    def add_backward_path(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1),tf.argmax(self.Y, 1)), tf.float32))
        # batch_norm requires update_ops
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    # end method add_backward_path


    def _W(self, name, shape):
        return tf.get_variable(name, shape, tf.float32, tf.contrib.layers.variance_scaling_initializer())
    # end method _W


    def _b(self, name, shape):
        return tf.get_variable(name, shape, tf.float32, tf.constant_initializer(0.1))
    # end method _b


    def fit(self, X, Y, val_data=None, n_epoch=10, batch_size=128, keep_prob=1.0, en_exp_decay=True,
            en_shuffle=True):
        if val_data is None:
            print("Train %d samples" % len(X))
        else:
            print("Train %d samples | Test %d samples" % (len(X), len(val_data[0])))
        log = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
        global_step = 0

        self.sess.run(tf.global_variables_initializer()) # initialize all variables
        for epoch in range(n_epoch):
            if en_shuffle:
                X, Y = sklearn.utils.shuffle(X, Y)
            local_step = 1
            
            for X_batch, Y_batch in zip(self.gen_batch(X, batch_size),
                                        self.gen_batch(Y, batch_size)): # batch training
                lr = self.decrease_lr(en_exp_decay, global_step, n_epoch, len(X), batch_size) 
                _, loss, acc = self.sess.run([self.train_op, self.loss, self.acc],
                                             {self.X: X_batch, self.Y: Y_batch,
                                              self.lr: lr,
                                              self.batch_size: len(X_batch),
                                              self.keep_prob: keep_prob,
                                              self.train_flag: True})
                local_step += 1
                global_step += 1
                if local_step % 50 == 0:
                    print ("Epoch %d/%d | Step %d/%d | train_loss: %.4f | train_acc: %.4f | lr: %.4f"
                        %(epoch+1, n_epoch, local_step, int(len(X)/batch_size), loss, acc, lr))

            if val_data is not None: # go through test dara, compute averaged validation loss and acc
                val_loss_list, val_acc_list = [], []
                for X_test_batch, Y_test_batch in zip(self.gen_batch(val_data[0], batch_size),
                                                      self.gen_batch(val_data[1], batch_size)):
                    v_loss, v_acc = self.sess.run([self.loss, self.acc],
                                                  {self.X: X_test_batch, self.Y: Y_test_batch,
                                                   self.batch_size: len(X_test_batch),
                                                   self.keep_prob: 1.0,
                                                   self.train_flag: False})
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
                    "test_loss: %.4f | test_acc: %.4f |" % (val_loss, val_acc),
                    "lr: %.4f" % (lr) )
        # end "for epoch in range(n_epoch):"

        return log
    # end method fit


    def predict(self, X_test, batch_size=128):
        batch_pred_list = []
        for X_test_batch in self.gen_batch(X_test, batch_size):
            batch_pred = self.sess.run(self.logits, {self.X: X_test_batch,
                                                     self.batch_size: len(X_test_batch),
                                                     self.keep_prob: 1.0,
                                                     self.train_flag: False})
            batch_pred_list.append(batch_pred)
        return np.vstack(batch_pred_list)
    # end method predict


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method gen_batch


    def decrease_lr(self, en_exp_decay, global_step, n_epoch, len_X, batch_size):
        if en_exp_decay:
            max_lr = 0.003
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