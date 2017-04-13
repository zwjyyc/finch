import tensorflow as tf
import numpy as np


class LinearSVMClassifier:
    def __init__(self, C, n_in, sess):
        self.C = C
        self.n_in = n_in
        self.sess = sess
        self.build_graph()
    # end constructor

    def build_graph(self):
        with tf.variable_scope('input_layer'):
            self.add_input_layer()
        with tf.name_scope('forward_path'):
            self.add_forward_path() 
        with tf.name_scope('output_layer'):
            self.add_output_layer()
        with tf.name_scope('backward_path'):
            self.add_backward_path()
    # end method build_graph


    def add_input_layer(self):
        self.batch_size = tf.placeholder(tf.int32)
        self.X = tf.placeholder(shape=(None, self.n_in), dtype=tf.float32)
        self.y = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.W = tf.Variable(tf.random_normal([self.n_in, 1]))
        self.b = tf.Variable(tf.constant(0.1, shape=[1]))
    # end method add_input_layer


    def add_forward_path(self):
        self.y_raw = tf.nn.bias_add(tf.matmul(self.X, self.W), self.b)
    # end method add_forward_path


    def add_output_layer(self):
        self.pred = tf.sign(self.y_raw)
    # end method add_output_layer


    def add_backward_path(self):
        regu_loss = 0.5 * tf.reduce_sum(tf.square(self.W))
        hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([self.batch_size,1]), 1-self.y*self.y_raw))
        self.loss = regu_loss + self.C * hinge_loss
        self.train_op = tf.train.GradientDescentOptimizer(1e-3).minimize(self.loss)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.y), tf.float32))
    # end method add_backward_path


    def fit(self, X, y, val_data, n_epoch=100, batch_size=100):
        print("Train %d samples | Test %d samples" % (len(X), len(val_data[0])))
        log = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
        
        self.sess.run(tf.global_variables_initializer()) # initialize all variables
        for epoch in range(n_epoch):
            for X_batch, y_batch in zip(self.gen_batch(X, batch_size), # batch training
                                        self.gen_batch(y, batch_size)):
                _, loss, acc = self.sess.run([self.train_op, self.loss, self.acc], feed_dict={self.X:X_batch,
                                              self.y:y_batch, self.batch_size:len(X_batch)})
            val_loss_list, val_acc_list = [], [] # compute validation loss and acc
            for X_test_batch, y_test_batch in zip(self.gen_batch(val_data[0], batch_size),
                                                  self.gen_batch(val_data[1], batch_size)):
                v_loss, v_acc = self.sess.run([self.loss, self.acc], feed_dict={self.X:X_test_batch,
                                               self.y:y_test_batch, self.batch_size:len(X_test_batch)})
                val_loss_list.append(v_loss)
                val_acc_list.append(v_acc)
            val_loss, val_acc = self.list_avg(val_loss_list), self.list_avg(val_acc_list)

            # append to log
            log['loss'].append(loss)
            log['acc'].append(acc)
            log['val_loss'].append(val_loss)
            log['val_acc'].append(val_acc)
            # verbose
            if epoch % 20 == 0:
                print ("%d / %d: train_loss: %.4f train_acc: %.4f | test_loss: %.4f test_acc: %.4f"
                    % (epoch+1, n_epoch, loss, acc, val_loss, val_acc))
            
        return log
    # end method fit


    def predict(self, X_test, batch_size=None):
        if batch_size is None:
            batch_size = len(X_test)
        batch_pred_list = []
        for X_test_batch in self.gen_batch(X_test, batch_size):
            batch_pred = self.sess.run(self.pred, feed_dict={self.X:X_test, self.batch_size:len(X_test_batch)})
            batch_pred_list.append(batch_pred)
        return np.concatenate(batch_pred_list)
    # end method predict


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method gen_batch


    def list_avg(self, l):
        return sum(l) / len(l)
    # end method list_avg
# end class LinearSVMClassifier
