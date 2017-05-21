import tensorflow as tf
import numpy as np


class LinearRegression:
    def __init__(self, n_in, l1_ratio=0.15, sess=tf.Session()):
        """
        Parameters:
        -----------
        l1_ratio: float
            l2_ratio = 1 - l1_ratio
        n_in: int
            Input dimensions
        sess: object
            tf.Session() object 
        """
        self.l1_ratio = l1_ratio
        self.n_in = n_in
        self.sess = sess
        self.build_graph()
    # end constructor

    def build_graph(self):
        self.add_input_layer()
        self.add_output_layer()
        self.add_backward_path()
    # end method build_graph


    def add_input_layer(self):
        self.X = tf.placeholder(shape=(None, self.n_in), dtype=tf.float32)
        self.Y = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.W = tf.Variable(tf.random_normal([self.n_in, 1]))
        self.b = tf.Variable(tf.constant(0.1, shape=[1]))
    # end method add_input_layer


    def add_output_layer(self):
        self.pred = tf.nn.bias_add(tf.matmul(self.X, self.W), self.b)
    # end method add_output_layer


    def add_backward_path(self):
        regr_loss = tf.reduce_mean(tf.square(self.pred - self.Y))
        l1_loss = tf.reduce_mean(tf.abs(self.W))
        l2_loss = tf.reduce_mean(tf.square(self.W))
        self.loss = regr_loss + self.l1_ratio * l1_loss + (1-self.l1_ratio) * l2_loss
        self.train_op = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)
    # end method add_backward_path


    def fit(self, X, Y, val_data, n_epoch=100, batch_size=100):
        print("Train %d samples | Test %d samples" % (len(X), len(val_data[0])))
        self.sess.run(tf.global_variables_initializer()) # initialize all variables
        for epoch in range(n_epoch):
            for X_batch, Y_batch in zip(self.gen_batch(X, batch_size), # batch training
                                        self.gen_batch(Y, batch_size)):
                _, loss= self.sess.run([self.train_op, self.loss], {self.X:X_batch, self.Y:Y_batch})

            val_loss_list = []
            for X_test_batch, Y_test_batch in zip(self.gen_batch(val_data[0], batch_size),
                                                  self.gen_batch(val_data[1], batch_size)):
                v_loss = self.sess.run(self.loss, {self.X:X_test_batch, self.Y:Y_test_batch})
                val_loss_list.append(v_loss)
            val_loss = self.list_avg(val_loss_list)

            print ("%d / %d: train_loss: %.4f | test_loss: %.4f" % (epoch+1, n_epoch, loss, val_loss))
    # end method fit


    def predict(self, X_test, batch_size=100):        
        batch_pred_list = []
        for X_test_batch in self.gen_batch(X_test, batch_size):
            batch_pred = self.sess.run(self.pred, {self.X:X_test_batch})
            batch_pred_list.append(batch_pred)
        return np.vstack(batch_pred_list)
    # end method predict


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method gen_batch


    def list_avg(self, l):
        return sum(l) / len(l)
    # end method list_avg
# end class