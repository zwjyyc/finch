import tensorflow as tf
import numpy as np
import math
import sklearn


class Conv1DClassifier:
    def __init__(self, seq_len, vocab_size, n_out, sess=tf.Session(),
                 embedding_dims=50, n_filters=250, kernel_size=3, padding='VALID'):
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
        self.padding = padding
        self.n_out = n_out
        self.sess = sess
        self.current_layer = None
        self.current_seq_len = self.seq_len
        self.build_graph()
    # end constructor
 
 
    def build_graph(self):
        self.add_input_layer()
        self.add_word_embedding()
        self.add_conv1d('conv1', filter_shape=[self.kernel_size, self.embedding_dims, self.n_filters])
        self.add_conv1d_highway('conv2', filter_shape=[self.kernel_size, self.n_filters, self.n_filters])
        self.add_global_maxpool()
        self.add_highway('highway')
        self.add_output_layer()   
        self.add_backward_path()
    # end method build_graph
 
 
    def add_input_layer(self):
        self.X = tf.placeholder(tf.int32, [None, self.seq_len])
        self.Y = tf.placeholder(tf.float32, [None, self.n_out])
        self.keep_prob = tf.placeholder(tf.float32)
        self.lr = tf.placeholder(tf.float32)
        self.current_layer = self.X
    # end method add_input_layer


    def add_word_embedding(self):
        E = tf.get_variable('E', [self.vocab_size,self.embedding_dims], tf.float32, tf.random_normal_initializer())
        E = tf.nn.embedding_lookup(E, self.current_layer)
        self.current_layer = tf.nn.dropout(E, self.keep_prob)
    # end method add_word_embedding_layer


    def add_conv1d(self, name, filter_shape, stride=1):
        W = self.call_W(name+'_w', filter_shape)
        b = self.call_b(name+'_b', [filter_shape[-1]])
        conv = tf.nn.conv1d(self.current_layer, W, stride=stride, padding=self.padding)
        conv = tf.nn.bias_add(conv, b)
        conv = tf.nn.relu(conv)
        self.current_layer = conv
        if self.padding == 'VALID':
            self.current_seq_len = int(self.current_seq_len - self.kernel_size + 1 / stride)
        if self.padding == 'SAME':
            self.current_seq_len = int(self.current_seq_len / stride)
    # end method add_conv1d_layer


    def add_conv1d_highway(self, name, filter_shape, stride=1, carry_bias=-1.0):
        X = self.current_layer

        W = self.call_W(name+'_w', filter_shape)
        b = tf.get_variable(name+'_b', filter_shape[-1], tf.float32, tf.constant_initializer(carry_bias))
        W_T = self.call_W(name+'_wt', filter_shape)
        b_T = self.call_b(name+'_bt', [filter_shape[-1]])

        H = tf.nn.relu(tf.nn.conv1d(X, W, stride, 'SAME') + b, name='activation')
        T = tf.sigmoid(tf.nn.conv1d(X, W_T, stride, 'SAME') + b_T, name='transform_gate')
        C = tf.subtract(1.0, T, name="carry_gate")

        self.current_layer = tf.add(tf.multiply(H, T), tf.multiply(X, C), 'y') # y = (H * T) + (x * C)
        self.current_seq_len = int(self.current_seq_len / stride)
    # end method add_conv1d_highway


    def add_global_maxpool(self):
        k = self.current_seq_len
        conv = tf.expand_dims(self.current_layer, 1)
        conv = tf.nn.max_pool(conv, ksize=[1,1,k,1], strides=[1,1,k,1], padding=self.padding)
        conv = tf.squeeze(conv)
        self.current_layer = conv
    # end method add_global_maxpool_layer


    def add_highway(self, name, carry_bias=-1.0):
        X = self.current_layer
        size = self.n_filters

        W_T = self.call_W(name+'_wt', [size, size])
        b_T = tf.get_variable(name+'_bt', [size], tf.float32, tf.constant_initializer(carry_bias))
        W = self.call_W(name+'_w', [size,size])
        b = self.call_b(name+'_b', [size])

        T = tf.sigmoid(tf.nn.bias_add(tf.matmul(X, W_T), b_T))
        H = tf.nn.relu(tf.nn.bias_add(tf.matmul(X, W), b))
        C = tf.subtract(1.0, T, name="carry_gate")

        Y = tf.add(tf.multiply(H, T), tf.multiply(X, C))
        self.current_layer = tf.reshape(Y, [-1, size])
    # end method add_highway


    def add_output_layer(self):
        in_dim = self.current_layer.get_shape()[1]
        W = self.call_W('logits_w', [in_dim, self.n_out])
        b = self.call_b('logits_b', [self.n_out])
        self.logits = tf.nn.bias_add(tf.matmul(self.current_layer, W), b)
    # end method add_output_layer


    def add_backward_path(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1),tf.argmax(self.Y, 1)), tf.float32))
    # end method add_backward_path


    def call_W(self, name, shape):
        return tf.get_variable(name, shape, tf.float32, tf.contrib.layers.variance_scaling_initializer())
    # end method _W


    def call_b(self, name, shape):
        return tf.get_variable(name, shape, tf.float32, tf.constant_initializer(0.01))
    # end method _b


    def fit(self, X, Y, val_data=None, n_epoch=10, batch_size=128, keep_prob=0.5, en_exp_decay=True,
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
                                             {self.X:X_batch, self.Y:Y_batch,
                                              self.lr:lr, self.keep_prob:keep_prob})
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
                                                  {self.X:X_test_batch, self.Y:Y_test_batch,
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
            batch_pred = self.sess.run(self.logits, {self.X:X_test_batch, self.keep_prob:1.0})
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