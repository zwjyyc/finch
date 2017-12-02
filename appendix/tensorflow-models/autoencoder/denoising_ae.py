import tensorflow as tf
import sklearn
import numpy as np
import math


class Autoencoder:
    def __init__(self, n_in, encoder_units, noise_level=0.5, sess=tf.Session()):
        self.sess = sess
        self.n_in = n_in
        self.encoder_units = encoder_units
        self.decoder_units = list(reversed(encoder_units))
        self.noise_level = noise_level
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.add_input_layer()
        self.add_encoders()
        self.add_decoders()
        self.add_backward_path()
    # end method build_graph


    def add_input_layer(self):
        self.X = tf.placeholder(tf.float32, [None, self.n_in])
        self.keep_prob = tf.placeholder(tf.float32)
        X_drop = tf.nn.dropout(self.X, self.keep_prob)
        self.X_noisy = X_drop + self.noise_level * tf.random_normal(tf.shape(X_drop))
    # end method add_input_layer


    def add_encoders(self):
        new_layer = self.X_noisy
        for unit in self.encoder_units[:-1]:
            new_layer = tf.layers.dense(new_layer, unit, tf.nn.relu)
        self.encoder_op = tf.layers.dense(new_layer, self.encoder_units[-1])
    # end method add_encoders


    def add_decoders(self):
        new_layer = self.encoder_op
        for unit in self.decoder_units[1:]:
            new_layer = tf.layers.dense(new_layer, unit, tf.nn.relu)
        self.logits = tf.layers.dense(new_layer, self.n_in)
        self.decoder_op = tf.sigmoid(self.logits)
    # end method add_decoders


    def add_backward_path(self):
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.X, logits=self.logits))
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
    # end method add_backward_path


    def fit(self, X_train, val_data, n_epoch=10, batch_size=128, en_shuffle=True, keep_prob=0.8):
        self.sess.run(tf.global_variables_initializer()) # initialize all variables
        global_step = 0
        for epoch in range(n_epoch):
            X_train = sklearn.utils.shuffle(X_train)
            for local_step, X_batch in enumerate(self.gen_batch(X_train, batch_size)):
                _, loss = self.sess.run([self.train_op, self.loss], {self.X:X_batch, self.keep_prob:keep_prob})
                if local_step % 100 == 0:
                    print ("Epoch %d/%d | Step %d/%d | train loss: %.4f"
                           %(epoch+1, n_epoch, local_step, len(X_train)//batch_size, loss))
                global_step += 1
            
            val_loss_list = []
            for X_test_batch in self.gen_batch(val_data, batch_size):
                v_loss = self.sess.run(self.loss, {self.X:X_test_batch, self.keep_prob:1.0})
                val_loss_list.append(v_loss)
            val_loss = sum(val_loss_list) / len(val_loss_list)
            print ("Epoch %d/%d | train loss: %.4f | test loss: %.4f" %(epoch+1, n_epoch, loss, v_loss))
    # end method fit_transform


    def transform(self, X_test, batch_size=128):
        res = []
        for X_batch in self.gen_batch(X_test, batch_size):
            res.append(self.sess.run(self.encoder_op, {self.X: X_batch, self.keep_prob:1.0}))
        return np.vstack(res)
    # end method transform


    def predict(self, X_test, batch_size=128):
        res = []
        for X_batch in self.gen_batch(X_test, batch_size):
            res.append(self.sess.run(self.decoder_op, {self.X: X_batch, self.keep_prob:1.0}))
        return np.vstack(res)
    # end method predict


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method gen_batch
# end class Autoencoder