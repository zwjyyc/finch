import tensorflow as tf
import numpy as np
import math


class Autoencoder:
    def __init__(self, sess, n_in, encoder_units, decoder_units):
        self.sess = sess
        self.n_in = n_in
        self.encoder_units = encoder_units
        self.decoder_units = decoder_units
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.X = tf.placeholder(tf.float32, [None, self.n_in])
        with tf.variable_scope('weights_tied') as scope:
            self.encoder_op = self.add_forward_path(self.X, self.encoder_units, 'encoder')
            scope.reuse_variables()
            self.decoder_op = self.add_forward_path(self.encoder_op, self.decoder_units, 'decoder')
        self.add_backward_path()
    # end method build_graph


    def add_forward_path(self, X, units, mode):
        new_layer = X
        if mode == 'encoder':
            forward = [self.n_in] + units
            names = ['layer%s'%i for i in range(len(forward)-1)]
        if mode == 'decoder':
            forward = units + [self.n_in]
            names = list(reversed(['layer%s'%i for i in range(len(forward)-1)]))
        for i in range(len(forward)-2):
            new_layer = self.fc(names[i], new_layer, forward[i], forward[i+1], mode)
            new_layer = tf.nn.relu(new_layer)
        new_layer = self.fc(names[-1], new_layer, forward[-2], forward[-1], mode)
        return new_layer
    # end method add_forward_path


    def add_backward_path(self):
        self.loss = tf.reduce_mean(tf.square(self.X - self.decoder_op))
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
    # end method add_backward_path


    def fc(self, name, X, fan_in, fan_out, mode):
        if mode == 'encoder':
            W = tf.get_variable(name+'_w', [fan_in,fan_out], tf.float32,
                                tf.contrib.layers.variance_scaling_initializer())
        if mode == 'decoder':
            W = tf.transpose(tf.get_variable(name+'_w'))
        b = tf.Variable(tf.constant(0.1, shape=[fan_out]))
        return tf.nn.bias_add(tf.matmul(X, W), b)
    # end method fc


    def fit(self, X_train, val_data, n_epoch=10, batch_size=128):
        self.sess.run(tf.global_variables_initializer()) # initialize all variables
        global_step = 0
        for epoch in range(n_epoch):
            # batch training
            for local_step, X_batch in enumerate(self.gen_batch(X_train, batch_size)):
                _, loss = self.sess.run([self.train_op, self.loss], {self.X:X_batch})
                if global_step == 0:
                    print("Initial loss: ", loss)
                if (local_step + 1) % 100 == 0:
                    print ("Epoch %d/%d | Step %d/%d | train loss: %.4f"
                           %(epoch+1, n_epoch, local_step+1, int(len(X_train)/batch_size), loss))
                global_step += 1
            
            val_loss_list = []
            for X_test_batch in self.gen_batch(val_data, batch_size):
                v_loss = self.sess.run(self.loss, {self.X:X_test_batch})
                val_loss_list.append(v_loss)
            val_loss = sum(val_loss_list) / len(val_loss_list)
            print ("Epoch %d/%d | train loss: %.4f | test loss: %.4f" %(epoch+1, n_epoch, loss, v_loss))
    # end method fit_transform


    def transform(self, X_test, batch_size=128):
        res = []
        for X_batch in self.gen_batch(X_test, batch_size):
            res.append(self.sess.run(self.encoder_op, {self.X: X_batch}))
        return np.concatenate(res)
    # end method transform


    def predict(self, X_test, batch_size=128):
        res = []
        for X_batch in self.gen_batch(X_test, batch_size):
            res.append(self.sess.run(self.decoder_op, {self.X: X_batch}))
        return np.concatenate(res)
    # end method predict


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method gen_batch
# end class Autoencoder
