import tensorflow as tf
import numpy as np
import math


class RNNClassifier:
    def __init__(self, n_in, n_step, n_hidden=128, n_out=2, n_layer=1):
        self.n_in = n_in
        self.n_step = n_step
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.n_layer = n_layer
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.X = tf.placeholder(tf.float32, [None, self.n_step, self.n_in])
        self.y = tf.placeholder(tf.float32, [None, self.n_out])
        self.W = {
            'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_out], stddev=math.sqrt(2.0/self.n_hidden)))
        }
        self.b = {
            'out': tf.Variable(tf.zeros([self.n_out]))
        }

        self.batch_size = tf.placeholder(tf.int32)
        self.lr = tf.placeholder(tf.float32)
        self.in_keep_prob = tf.placeholder(tf.float32)
        self.out_keep_prob = tf.placeholder(tf.float32)

        self.pred = self.rnn(self.X, self.W, self.b, self.in_keep_prob, self.out_keep_prob)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.acc = tf.reduce_mean( tf.cast( tf.equal( tf.argmax(self.pred, 1), tf.argmax(self.y, 1) ), tf.float32 ) )

        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
    # end method build_graph


    def rnn(self, X, W, b, in_keep_prob, out_keep_prob):
        cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)
        cell = tf.contrib.rnn.DropoutWrapper(cell, in_keep_prob, out_keep_prob)
        cells = tf.contrib.rnn.MultiRNNCell([cell] * self.n_layer)        
        outputs, final_state = tf.nn.dynamic_rnn(cells, X, time_major=False, dtype=tf.float32)

        # (batch, n_step, n_hidden) -> (n_step, batch, n_hidden) -> n_step * [(batch, n_hidden)]
        outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
        results = tf.nn.bias_add(tf.matmul(outputs[-1], W['out']), b['out'])
        return results
    # end method rnn


    def fit(self, X, y, val_data=None, n_epoch=10, batch_size=32, en_exp_decay=True, keep_prob_tuple=(1.0, 1.0)):
        if val_data is None:
            print("Train %d samples" % len(X) )
        else:
            print("Train %d samples | Test %d samples" % (len(X), len(val_data[0])))
        log = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
        global_step = 0

        self.sess.run(self.init) # initialize all variables

        for epoch in range(n_epoch):
            # batch training
            i = 0
            for X_train_batch, y_train_batch in zip(self.gen_batch(X, batch_size), self.gen_batch(y, batch_size)):
                lr = self.get_lr(en_exp_decay, global_step, n_epoch, len(X), batch_size)             
                self.sess.run(self.train_op, feed_dict = self.get_train_op_dict(X_train_batch, y_train_batch,
                                             batch_size, lr, keep_prob_tuple))
                i += 1
                global_step += 1
                # compute training loss and acc
                loss, acc = self.sess.run([self.loss, self.acc], feed_dict = self.get_val_dict(X_train_batch,
                                           y_train_batch, batch_size))
                if (i+1) % 100 == 0:
                    print ('Epoch %d/%d | Step %d/%d | train loss: %.4f | train acc: %.4f | lr: %.4f'
                           %(epoch+1, n_epoch, i+1, int(len(X)/batch_size), loss, acc, lr))
            if val_data is not None:
                # go through testing data, average validation loss and acc
                val_loss_list, val_acc_list = [], []
                for X_test_batch, y_test_batch in zip(self.gen_batch(val_data[0], batch_size),
                                                    self.gen_batch(val_data[1], batch_size)):
                    v_loss, v_acc = self.sess.run([self.loss, self.acc], feed_dict = self.get_val_dict(
                                                X_test_batch, y_test_batch, batch_size))
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
                print ("Epoch %d/%d | train loss: %.4f | train acc: %.4f |" % (epoch+1, n_epoch, loss, acc),
                       "lr: %.4f" % (lr) )
            else:
                print ("Epoch %d/%d | train loss: %.4f | train acc: %.4f |" % (epoch+1, n_epoch, loss, acc),
                       "test loss: %.4f | test acc: %.4f |" % (val_loss, val_acc),
                       "lr: %.4f" % (lr) )
        return log
    # end method fit


    def predict(self, X_test, batch_size=32):
        batch_pred_list = []
        for X_test_batch in self.gen_batch(X_test, batch_size):
            batch_pred = self.sess.run(self.pred, feed_dict={self.X: X_test_batch, self.batch_size: batch_size,
                                                             self.in_keep_prob: 1.0, self.out_keep_prob: 1.0})
            batch_pred_list.append(batch_pred)
        return np.concatenate(batch_pred_list)
    # end method predict


    def close(self):
        self.sess.close()
        tf.reset_default_graph()
    # end method close


    def gen_batch(self, arr, batch_size):
        if len(arr) % batch_size != 0:
            new_len = len(arr) - len(arr) % batch_size
            for i in range(0, new_len, batch_size):
                yield arr[i : i + batch_size]
        else:
            for i in range(0, len(arr), batch_size):
                yield arr[i : i + batch_size]
    # end method gen_batch


    def get_lr(self, en_exp_decay, global_step, n_epoch, len_X, batch_size):
        if en_exp_decay:
            max_lr = 0.003
            min_lr = 0.0001
            decay_rate = math.log(min_lr/max_lr) / (-n_epoch*len_X/batch_size)
            lr = max_lr*math.exp(-decay_rate*global_step)
        else:
            lr = 0.001
        return lr
    # end method get_lr


    def get_train_op_dict(self, X_batch, y_batch, batch_size, lr, keep_prob_tuple):
        return {self.X: X_batch, self.y: y_batch, self.batch_size: batch_size, self.lr: lr,
                self.in_keep_prob: keep_prob_tuple[0], self.out_keep_prob: keep_prob_tuple[1]}
    # end method get_train_op_dict


    def get_val_dict(self, X_batch, y_batch, batch_size):
        return {self.X: X_batch, self.y: y_batch, self.batch_size: batch_size, self.in_keep_prob: 1.0,
                self.out_keep_prob: 1.0}
    # end method get_val_dict


    def list_avg(self, l):
        return sum(l) / len(l)
    # end method list_avg
# end class LinearSVMClassifier
