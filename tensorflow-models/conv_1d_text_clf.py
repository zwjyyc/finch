import tensorflow as tf
import numpy as np
import math
import sklearn


class Conv1DClassifier:
    def __init__(self, seq_len, vocab_size, embedding_dims, n_filters, kernel_size, hidden_dims, n_out, sess):
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
        hidden_dims: int
            Ouput dimensions of the fully-connected layer
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
        self.hidden_dims = hidden_dims
        self.n_out = n_out
        self.sess = sess
        self.build_graph()
    # end constructor
 
 
    def build_graph(self):
        with tf.name_scope('input_layer'):
            self.add_input_layer()
        with tf.variable_scope('forward_path'):
            self.add_word_embedding_layer()
            self.add_conv1d_layer('conv', filter_shape=[self.kernel_size, self.embedding_dims, self.n_filters])
            self.add_global_maxpool_layer( self.seq_len-self.kernel_size+1 )
            self.add_fc_layer('fully_connected', [self.n_filters, self.hidden_dims])
        with tf.variable_scope('output_layer'):
            self.add_output_layer(in_dim=self.hidden_dims)   
        with tf.name_scope('backward_path'):
            self.add_backward_path()
    # end method build_graph
 
 
    def add_input_layer(self):
        self.X = tf.placeholder(tf.int32, [None, self.seq_len])
        self.y = tf.placeholder(tf.float32, [None, self.n_out])
        self.keep_prob = tf.placeholder(tf.float32)
    # end method add_input_layer


    def add_word_embedding_layer(self):
        embedding_mat = tf.get_variable('embedding_mat', [self.vocab_size,self.embedding_dims], tf.float32,
                                        tf.random_normal_initializer())
        self.embedding_out = tf.nn.embedding_lookup(embedding_mat, self.X)
    # end method add_word_embedding_layer


    def add_conv1d_layer(self, name, filter_shape):
        self.conv = self.conv1d_wrapper(self.embedding_out, self._W(name+'_w', filter_shape),
                                        self._b(name+'_b', [filter_shape[-1]]))
    # end method add_conv1d_layer


    def conv1d_wrapper(self, X, W, b, stride=1):
        conv = tf.nn.conv1d(X, W, stride=stride, padding='VALID')
        conv = tf.nn.bias_add(conv, b)
        conv = tf.nn.relu(conv)
        return conv
    # end method conv1d_wrapper


    def add_global_maxpool_layer(self, k):
        conv = tf.expand_dims(self.conv, 1)
        conv = tf.nn.max_pool(conv, ksize=[1,1,k,1], strides=[1,1,k,1], padding='VALID')
        self.conv = tf.squeeze(conv)
    # end method add_global_maxpool_layer


    def add_fc_layer(self, name, w_shape):
        W = self._W(name+'_w', w_shape)
        b = self._b(name+'_b', [w_shape[-1]])
        fc = tf.nn.bias_add(tf.matmul(self.conv, W), b)
        fc = tf.nn.relu(fc)
        self.conv = tf.nn.dropout(fc, self.keep_prob)
    # end method add_fc_layer


    def add_output_layer(self, in_dim):
        self.logits = tf.nn.bias_add(tf.matmul(self.conv, self._W('w_out', [in_dim,self.n_out])),
                                     self._b('b_out', [self.n_out]))
    # end method add_output_layer


    def add_backward_path(self):
        self.lr = tf.placeholder(tf.float32)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1),tf.argmax(self.y, 1)), tf.float32))
    # end method add_backward_path


    def _W(self, name, shape):
        return tf.get_variable(name, shape, tf.float32, tf.truncated_normal_initializer(stddev=0.1))
    # end method _W


    def _b(self, name, shape):
        return tf.get_variable(name, shape, tf.float32, tf.constant_initializer(0.1))
    # end method _b


    def fit(self, X, y, val_data=None, n_epoch=10, batch_size=128, keep_prob=0.5, en_exp_decay=True,
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
                X_train, y_train = sklearn.utils.shuffle(X, y)
            else:
                X_train, y_train = X, y
            local_step = 1
            
            for X_batch, y_batch in zip(self.gen_batch(X_train,batch_size),
                                        self.gen_batch(y_train,batch_size)): # batch training
                lr = self.decrease_lr(en_exp_decay, global_step, n_epoch, len(X), batch_size) 
                _, loss, acc = self.sess.run([self.train_op, self.loss, self.acc],
                                              feed_dict={self.X:X_batch, self.y:y_batch,
                                                         self.lr:lr, self.keep_prob:keep_prob})
                local_step += 1
                global_step += 1
                if local_step % 50 == 0:
                    print ("Epoch %d/%d | Step %d/%d | train_loss: %.4f | train_acc: %.4f | lr: %.4f"
                        %(epoch+1, n_epoch, local_step, int(len(X)/batch_size), loss, acc, lr))

            if val_data is not None: # go through test dara, compute averaged validation loss and acc
                val_loss_list, val_acc_list = [], []
                for X_test_batch, y_test_batch in zip(self.gen_batch(val_data[0], batch_size),
                                                      self.gen_batch(val_data[1], batch_size)):
                    v_loss, v_acc = self.sess.run([self.loss, self.acc],
                                                   feed_dict={self.X:X_test_batch, self.y:y_test_batch,
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
                    "test_loss: %.4f | test_acc: %.4f |" % (val_loss, val_acc),
                    "lr: %.4f" % (lr) )
        # end "for epoch in range(n_epoch):"

        return log
    # end method fit


    def predict(self, X_test, batch_size=128):
        batch_pred_list = []
        for X_test_batch in self.gen_batch(X_test, batch_size):
            batch_pred = self.sess.run(self.logits, feed_dict={self.X:X_test_batch, self.keep_prob:1.0})
            batch_pred_list.append(batch_pred)
        return np.concatenate(batch_pred_list)
    # end method predict


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method gen_batch


    def decrease_lr(self, en_exp_decay, global_step, n_epoch, len_X, batch_size):
        if en_exp_decay:
            max_lr = 0.003
            min_lr = 0.0001
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