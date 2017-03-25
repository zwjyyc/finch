import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
import numpy as np
import math


class MLPClassifier:
    def __init__(self, n_in, n_hidden_list, n_out=2):
        self.n_in = n_in
        self.n_hid = n_hidden_list
        self.n_out = n_out

        self.build_graph()
    # end constructor


    def build_graph(self):
        self.X = tf.placeholder(tf.float32, [None, self.n_in])
        self.y = tf.placeholder(tf.float32, [None, self.n_out])

        self.lr = tf.placeholder(tf.float32)

        self.pred = self.mlp(self.X)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
        self.train = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.acc = tf.reduce_mean( tf.cast( tf.equal( tf.argmax(self.pred, 1), tf.argmax(self.y, 1) ), tf.float32 ) )

        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
    # end method build_graph

    def mlp(self, X):
        # [n_samples, n_feature] dot [n_feature, n_hidden[0]] -> [n_samples, n_hidden[0]]
        new_layer = self.lin_equ(X, self.get_W(self.n_in, self.n_hid[0]), self.get_b(self.n_hid[0]))
        new_layer = tf.nn.relu(batch_norm(new_layer))
        """
        if there are three hidden layers: [1, 2, 3], we need two iterations:
        [n_samples, 1] dot [1, 2] -> [n_samples, 2]
        [n_samples, 2] dot [2, 3] -> [n_samples, 3]
        finally we have [n_samples, n_hidden[-1]]
        """
        if len(self.n_hid) != 1:
            for idx in range(len(self.n_hid) - 1):
                new_layer = self.lin_equ(new_layer, self.get_W(self.n_hid[idx], self.n_hid[idx+1]),
                                         self.get_b(self.n_hid[idx+1]))
                new_layer = tf.nn.relu(batch_norm(new_layer))
        # [n_samples, n_hidden[-1]] dot [n_hidden[-1], n_out] -> [n_samples, n_out]
        out_layer = self.lin_equ(new_layer, self.get_W(self.n_hid[-1], self.n_out), self.get_b(self.n_out))
        return out_layer
    # end method mlp


    def get_W(self, fan_in, fan_out):
        return tf.Variable(tf.random_normal([fan_in, fan_out], stddev=math.sqrt(2.0 / fan_in)))
    # end method get_W


    def get_b(self, fan_out):
        return tf.Variable(tf.zeros([fan_out]))
    # end method get_b


    def lin_equ(self, X, W, b):
        return tf.nn.bias_add(tf.matmul(X, W), b)
    # end method get_equ


    def fit(self, X, y, validation_data, n_epoch=10, batch_size=32, en_exp_decay=True):
        print("Train %d samples | Test %d samples" % (len(X), len(validation_data[0])))
        log = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
        global_step = 0

        self.sess.run(self.init) # initialize all variables

        for epoch in range(n_epoch):
            # batch training
            for X_batch, y_batch in zip(self.gen_batch(X, batch_size), self.gen_batch(y, batch_size)):

                if en_exp_decay:
                    max_lr = 0.003
                    min_lr = 0.0001
                    decay_rate = math.log(min_lr/max_lr) / (-n_epoch*len(X)/batch_size)
                    lr = max_lr*math.exp(-decay_rate*global_step)
                else:
                    lr = 0.001

                self.sess.run(self.train, feed_dict={self.X: X_batch, self.y: y_batch, self.lr: lr})
                global_step += 1
            
            # compute training loss and acc
            loss, acc = self.sess.run([self.loss, self.acc], feed_dict={self.X: X_batch, self.y: y_batch})
            # compute validation loss and acc
            val_loss_list, val_acc_list = [], []
            for X_test_batch, y_test_batch in zip(self.gen_batch(validation_data[0], batch_size),
                                                  self.gen_batch(validation_data[1], batch_size)):
                val_loss_list.append(self.sess.run(self.loss, feed_dict={self.X: X_test_batch,
                                                                         self.y: y_test_batch}))
                val_acc_list.append(self.sess.run(self.acc, feed_dict={self.X: X_test_batch,
                                                                       self.y: y_test_batch}))
            val_loss, val_acc = sum(val_loss_list)/len(val_loss_list), sum(val_acc_list)/len(val_acc_list)

            # append to log
            log['loss'].append(loss)
            log['acc'].append(acc)
            log['val_loss'].append(val_loss)
            log['val_acc'].append(val_acc)

            # verbose
            print ("%d / %d: train_loss: %.4f train_acc: %.4f |" % (epoch+1, n_epoch, loss, acc),
                   "test_loss: %.4f test_acc: %.4f |" % (val_loss, val_acc),
                   "learning rate: %.4f" % (lr) )
        return log
    # end method fit


    def predict(self, X_test, batch_size=32):
        batch_pred_list = []
        for X_test_batch in self.gen_batch(X_test, batch_size):
            batch_pred = self.sess.run(self.pred, feed_dict={self.X: X_test_batch})
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
