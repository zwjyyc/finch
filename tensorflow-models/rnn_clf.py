import tensorflow as tf
import numpy as np
import math


class RNNClassifier:
    def __init__(self, n_in, n_step, n_hidden=128, n_out=2):
        self.n_in = n_in
        self.n_step = n_step
        self.n_hidden = n_hidden
        self.n_out = n_out

        self.build_graph()
    # end constructor


    def build_graph(self):
        self.X = tf.placeholder(tf.float32, [None, self.n_step, self.n_in])
        self.y = tf.placeholder(tf.float32, [None, self.n_out])

        self.W = {
            'in': tf.Variable(tf.random_normal([self.n_in, self.n_hidden], stddev=math.sqrt(2.0/self.n_in))),
            'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_out], stddev=math.sqrt(2.0/self.n_hidden)))
        }
        self.b = {
            'in': tf.Variable(tf.zeros([self.n_hidden])),
            'out': tf.Variable(tf.zeros([self.n_out]))
        }

        self.batch_size = tf.placeholder(tf.int32)
        self.lr = tf.placeholder(tf.float32)

        self.pred = self.rnn(self.X, self.W, self.b)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
        self.train = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.acc = tf.reduce_mean( tf.cast( tf.equal( tf.argmax(self.pred, 1), tf.argmax(self.y, 1) ), tf.float32 ) )

        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
    # end method build_graph


    def rnn(self, X, W, b):
        X = tf.reshape(X, [-1, self.n_in])
        X_in = tf.matmul(X, W['in']) + b['in']
        X_in = tf.reshape(X_in, [-1, self.n_step, self.n_hidden])

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)
        init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
        
        outputs = tf.unstack(tf.transpose(outputs, [1,0,2])) # (batch, n_step, n_out) -> n_step * [(batch, n_out)]
        results = tf.nn.bias_add(tf.matmul(outputs[-1], W['out']), b['out'])

        return results
    # end method rnn

    def fit(self, X, y, validation_data, n_epoch=10, batch_size=32):
        print("Train %d samples | Test %d samples" % (len(X), len(validation_data[0])))
        log = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
        max_lr = 0.003
        min_lr = 0.0001
        decay_rate = math.log(min_lr/max_lr) / (-n_epoch*len(X)/batch_size)

        self.sess.run(self.init) # initialize all variables

        global_step = 0
        for epoch in range(n_epoch):
            # batch training
            for X_batch, y_batch in zip(self.gen_batch(X, batch_size), self.gen_batch(y, batch_size)):
                self.sess.run(self.train, feed_dict={self.X: X_batch, self.y: y_batch,
                                                     self.batch_size: batch_size,
                                                     self.lr: max_lr*math.exp(-decay_rate*global_step)})
                global_step += 1
            # compute training loss and acc
            loss, acc = self.sess.run([self.loss, self.acc], feed_dict={self.X: X_batch, self.y: y_batch,
                                                                        self.batch_size: batch_size})
            # compute validation loss and acc
            val_loss_list, val_acc_list = [], []
            for X_test_batch, y_test_batch in zip(self.gen_batch(validation_data[0], batch_size),
                                                  self.gen_batch(validation_data[1], batch_size)):
                val_loss_list.append(self.sess.run(self.loss, feed_dict={self.X: X_test_batch, self.y: y_test_batch,
                                     self.batch_size: batch_size}))
                val_acc_list.append(self.sess.run(self.acc, feed_dict={self.X: X_test_batch, self.y: y_test_batch,
                                    self.batch_size: batch_size}))
            val_loss, val_acc = sum(val_loss_list)/len(val_loss_list), sum(val_acc_list)/len(val_acc_list)

            # append to log
            log['loss'].append(loss)
            log['acc'].append(acc)
            log['val_loss'].append(val_loss)
            log['val_acc'].append(val_acc)

            # verbose
            print ("%d / %d: train_loss: %.4f train_acc: %.4f | test_loss: %.4f test_acc: %.4f"
                   % (epoch+1, n_epoch, loss, acc, val_loss, val_acc))
            
        return log
    # end method fit

    def predict(self, X_test, batch_size=32):
        batch_pred_list = []
        for X_test_batch in self.gen_batch(X_test, batch_size):
            batch_pred = self.sess.run(self.pred, feed_dict={self.X: X_test_batch, self.batch_size: batch_size})
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
# end class LinearSVMClassifier
