import tensorflow as tf
import numpy as np
import math
import sklearn


class RNNClassifier:
    def __init__(self, n_in, n_step, n_hidden, n_out, n_layer, sess, stateful=False):
        self.n_in = n_in
        self.n_step = n_step
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.n_layer = n_layer
        self.sess = sess
        self.stateful = stateful
        self.build_graph()
    # end constructor


    def build_graph(self):
        with tf.variable_scope('input_layer'):
            self.add_input_layer()
        with tf.name_scope('forward_path'):
            self.add_lstm_cells()
            self.add_dynamic_rnn()
            self.reshape_rnn_out()
        with tf.name_scope('output_layer'):
            self.add_output_layer()
        with tf.name_scope('backward_path'):
            self.add_backward_path()
    # end method build_graph


    def add_input_layer(self):
        self.batch_size = tf.placeholder(tf.int32)
        self.X = tf.placeholder(tf.float32, [None, self.n_step, self.n_in])
        self.y = tf.placeholder(tf.float32, [None, self.n_out])
        self.W = tf.get_variable('W', [self.n_hidden, self.n_out], tf.float32,
                                 tf.contrib.layers.variance_scaling_initializer())
        self.b = tf.get_variable('b', [self.n_out], tf.float32, tf.constant_initializer(0.0))
        self.in_keep_prob = tf.placeholder(tf.float32)
        self.out_keep_prob = tf.placeholder(tf.float32)
    # end method add_input_layer


    def add_lstm_cells(self):
        cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)
        cell = tf.contrib.rnn.DropoutWrapper(cell, self.in_keep_prob, self.out_keep_prob)
        self.cells = tf.contrib.rnn.MultiRNNCell([cell] * self.n_layer)
    # end method add_rnn_cells


    def add_dynamic_rnn(self):
        self.init_state = self.cells.zero_state(self.batch_size, tf.float32)        
        self.rnn_out, self.final_state = tf.nn.dynamic_rnn(self.cells, self.X, initial_state=self.init_state,
                                                           time_major=False)
    # end method add_dynamic_rnn


    def reshape_rnn_out(self):
        # (batch, n_step, n_hidden) -> (n_step, batch, n_hidden) -> n_step * [(batch, n_hidden)]
        self.rnn_out = tf.unstack(tf.transpose(self.rnn_out, [1,0,2]))
    # end method add_rnn_out


    def add_output_layer(self):
        self.logits = tf.nn.bias_add(tf.matmul(self.rnn_out[-1], self.W), self.b)
    # end method add_output_layer


    def add_backward_path(self):
        self.lr = tf.placeholder(tf.float32)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits,1),tf.argmax(self.y,1)), tf.float32))
    # end method add_backward_path


    def fit(self, X, y, val_data=None, n_epoch=10, batch_size=128, en_exp_decay=True, en_shuffle=True, 
            keep_prob_tuple=(1.0,1.0)):
        if val_data is None:
            print("Train %d samples" % len(X) )
        else:
            print("Train %d samples | Test %d samples" % (len(X), len(val_data[0])))
        log = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
        global_step = 0

        self.sess.run(tf.global_variables_initializer()) # initialize all variables
        for epoch in range(n_epoch): # batch training

            if en_shuffle:
                X_train, y_train = sklearn.utils.shuffle(X, y)
            else:
                X_train, y_train = X, y
            local_step = 1
            next_state = self.sess.run(self.init_state, feed_dict={self.batch_size:batch_size})

            for X_train_batch, y_train_batch in zip(self.gen_batch(X_train, batch_size),
                                                    self.gen_batch(y_train, batch_size)):
                lr = self.decrease_lr(en_exp_decay, global_step, n_epoch, len(X), batch_size)
                if self.stateful:
                    _, next_state, loss, acc = self.sess.run([self.train_op, self.final_state, self.loss, self.acc],
                        feed_dict = {self.X:X_train_batch, self.y:y_train_batch, self.batch_size:batch_size,
                            self.lr:lr, self.in_keep_prob:keep_prob_tuple[0], self.out_keep_prob:keep_prob_tuple[1],
                                self.init_state:next_state})
                else:             
                    _, loss, acc = self.sess.run([self.train_op, self.loss, self.acc],
                        feed_dict = {self.X:X_train_batch, self.y:y_train_batch, self.batch_size:batch_size,
                            self.lr:lr, self.in_keep_prob:keep_prob_tuple[0], self.out_keep_prob:keep_prob_tuple[1]})
                local_step += 1
                global_step += 1
                if local_step % 50 == 0:
                    print ('Epoch %d/%d | Step %d/%d | train_loss: %.4f | train_acc: %.4f | lr: %.4f'
                           %(epoch+1, n_epoch, local_step, int(len(X)/batch_size), loss, acc, lr))

            if val_data is not None: # go through testing data, average validation loss and ac 
                val_loss_list, val_acc_list = [], []
                next_state = self.sess.run(self.init_state, feed_dict={self.batch_size:batch_size})
                for X_test_batch, y_test_batch in zip(self.gen_batch(val_data[0], batch_size),
                                                      self.gen_batch(val_data[1], batch_size)):
                    if self.stateful:
                        v_loss, v_acc, next_state = self.sess.run([self.loss, self.acc, self.final_state],
                                feed_dict = {self.X:X_test_batch, self.y:y_test_batch, self.batch_size:batch_size,
                                    self.in_keep_prob:1.0, self.out_keep_prob:1.0, self.init_state:next_state})
                    else:
                        v_loss, v_acc = self.sess.run([self.loss, self.acc], feed_dict = {self.X:X_test_batch,
                            self.y:y_test_batch, self.batch_size:batch_size, self.in_keep_prob:1.0,
                                self.out_keep_prob:1.0})
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
        next_state = self.sess.run(self.init_state, feed_dict={self.batch_size:batch_size})
        for X_test_batch in self.gen_batch(X_test, batch_size, is_size_equal=False):
            if (self.stateful) and (len(X_test_batch) == batch_size):
                batch_pred, next_state = self.sess.run([self.logits, self.final_state], 
                    feed_dict = {self.X:X_test_batch, self.batch_size:batch_size, self.in_keep_prob:1.0,
                        self.out_keep_prob:1.0, self.init_state:next_state})
            else:
                batch_pred = self.sess.run(self.logits, feed_dict = {self.X:X_test_batch,
                    self.batch_size:len(X_test_batch), self.in_keep_prob:1.0, self.out_keep_prob:1.0})
            batch_pred_list.append(batch_pred)
        return np.concatenate(batch_pred_list)
    # end method predict


    def gen_batch(self, arr, batch_size, is_size_equal=True):
        if is_size_equal:
            if len(arr) % batch_size != 0:
                new_len = len(arr) - len(arr) % batch_size
                for i in range(0, new_len, batch_size):
                    yield arr[i : i+batch_size]
            else:
                for i in range(0, len(arr), batch_size):
                    yield arr[i : i+batch_size]
        else:
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
# end class LinearSVMClassifier
