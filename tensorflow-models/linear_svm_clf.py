import tensorflow as tf
import numpy as np


class LinearSVMClassifier:
    def __init__(self, C, n_in):
        self.C = C
        self.n_in = n_in

        self.build_graph()
    # end constructor

    def build_graph(self):
        self.X = tf.placeholder(shape=(None, self.n_in), dtype=tf.float32)
        self.y = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        self.W = tf.Variable(tf.random_normal(shape=(self.n_in, 1)))
        self.b = tf.Variable(tf.constant(0.1, shape=[1]))
        self.batch_size = tf.placeholder(tf.int32)

        y_raw = tf.matmul(self.X, self.W) + self.b
        regu_loss = 0.5 * tf.reduce_sum(tf.square(self.W))
        hinge_loss = tf.reduce_sum( tf.maximum( tf.zeros([self.batch_size, 1]), 1 - self.y * y_raw ) )

        self.pred = tf.sign(y_raw)
        self.loss = regu_loss + self.C * hinge_loss
        self.train = tf.train.GradientDescentOptimizer(1e-3).minimize(self.loss)
        self.acc = tf.reduce_mean( tf.cast( tf.equal(self.pred, self.y), tf.float32 ) )

        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
    # end method build_graph

    def fit(self, X, y, validation_data, n_epoch=10, batch_size=None):
        print("Train %d samples | Test %d samples" % (len(X), len(validation_data[0])))
        log = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}

        test_batch_size = batch_size
        if batch_size is None:
            batch_size = len(X)
            test_batch_size = len(validation_data[0])
        
        self.sess.run(self.init) # initialize all variables

        for epoch in range(n_epoch):
            # batch training
            for X_batch, y_batch in zip(self.gen_batch(X, batch_size), self.gen_batch(y, batch_size)):
                self.sess.run(self.train, feed_dict={self.X: X_batch, self.y: y_batch,
                                                     self.batch_size: batch_size})
            # compute training loss and acc
            loss, acc = self.sess.run([self.loss, self.acc], feed_dict={self.X: X_batch, self.y: y_batch,
                                                                        self.batch_size: batch_size})
            # compute validation loss and acc
            val_loss_list, val_acc_list = [], []
            for X_test_batch, y_test_batch in zip(self.gen_batch(validation_data[0], test_batch_size),
                                                  self.gen_batch(validation_data[1], test_batch_size)):
                val_loss_list.append(self.sess.run(self.loss, feed_dict={self.X: X_test_batch, self.y: y_test_batch,
                                     self.batch_size: test_batch_size}))
                val_acc_list.append(self.sess.run(self.acc, feed_dict={self.X: X_test_batch, self.y: y_test_batch,
                                    self.batch_size: test_batch_size}))
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

    def predict(self, X_test, batch_size=None):
        if batch_size is None:
            batch_size = len(X_test)
        batch_pred_list = []
        for X_test_batch in self.gen_batch(X_test, batch_size):
            batch_pred = self.sess.run(self.pred, feed_dict={self.X: X_test, self.batch_size: batch_size})
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
