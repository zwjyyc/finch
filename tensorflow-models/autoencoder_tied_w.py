import tensorflow as tf
import numpy as np
import math


class Autoencoder:
    def __init__(self, n_in, encoder_units, decoder_units):
        self.n_in = n_in
        self.encoder_units = encoder_units
        self.decoder_units = decoder_units
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.X = tf.placeholder(tf.float32, [None, self.n_in])
        with tf.variable_scope('forward_path') as scope:
            self.encoder_op = self.encoder(self.X, self.encoder_units)
            scope.reuse_variables()
            self.decoder_op = self.decoder(self.encoder_op, self.decoder_units)
        self.loss = tf.reduce_mean(tf.square(self.X - self.decoder_op))
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
    # end method build_graph


    def encoder(self, X, encoder_units):
        new_layer = X
        forward = [self.n_in] + encoder_units
        names = ['layer%s'%i for i in range(len(forward)-1)]
        for i in range(len(forward)-2):
            new_layer = self.fc(names[i], new_layer, forward[i], forward[i+1])
            new_layer = tf.nn.relu(new_layer)
        new_layer = self.fc(names[-1], new_layer, forward[-2], forward[-1])
        return new_layer
    # end method encoder


    def decoder(self, X, decoder_units):
        new_layer = X
        forward = decoder_units + [self.n_in]
        names = list(reversed(['layer%s'%i for i in range(len(forward)-1)]))
        for i in range(len(forward)-2):
            new_layer = self._fc(names[i], new_layer, forward[i], forward[i+1])
            new_layer = tf.nn.relu(new_layer)
        new_layer = self._fc(names[-1], new_layer, forward[-2], forward[-1])
        return new_layer
    # end method decoder


    def fc(self, name, X, fan_in, fan_out):
        W = tf.get_variable(name+'_w', [fan_in,fan_out], tf.float32,
                            tf.contrib.layers.variance_scaling_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[fan_out]))
        return tf.nn.bias_add(tf.matmul(X, W), b)
    # end method fc


    def _fc(self, name, X, fan_in, fan_out):
        W = tf.transpose(tf.get_variable(name+'_w'))
        b = tf.Variable(tf.constant(0.1, shape=[fan_out]))
        return tf.nn.bias_add(tf.matmul(X, W), b)
    # end method _fc


    def fit_transform(self, X_train, n_epoch=10, batch_size=32):
        self.sess.run(self.init) # initialize all variables
        global_step = 0
        for epoch in range(n_epoch):
            # batch training
            for local_step, X_batch in enumerate(self.gen_batch(X_train, batch_size)):
                _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.X: X_batch})
                if global_step == 0:
                    print("Initial loss: ", loss)
                if (local_step + 1) % 100 == 0:
                    print ("Epoch %d/%d | Step %d/%d | train loss: %.4f"
                           %(epoch+1, n_epoch, local_step+1, int(len(X_train)/batch_size), loss))
                global_step += 1

        res = []
        for X_batch in self.gen_batch(X_train, batch_size):
            res.append(self.sess.run(self.encoder_op, feed_dict={self.X: X_batch}))
        return np.concatenate(res)
    # end method fit_transform


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method gen_batch
# end class Autoencoder
