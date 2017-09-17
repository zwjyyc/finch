import tensorflow as tf
import sklearn
import numpy as np
import math


class Autoencoder:
    def __init__(self, n_in, encoder_units, sess=tf.Session()):
        self.sess = sess
        self.n_in = n_in
        self.encoder_units = encoder_units
        self.decoder_units = list(reversed(encoder_units))
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
    # end method add_input_layer


    def add_encoders(self):
        new_layer = self.X
        for unit in self.encoder_units[:-1]:
            new_layer = tf.layers.dense(new_layer, unit, tf.nn.elu)
        self.mean = tf.layers.dense(new_layer, self.encoder_units[-1])
        self.gamma = tf.layers.dense(new_layer, self.encoder_units[-1])
        noise = tf.random_normal(tf.shape(self.gamma))
        self.encoder_op = self.mean + tf.exp(0.5 * self.gamma) * noise
    # end method add_encoders


    def add_decoders(self):
        new_layer = self.encoder_op
        for unit in self.decoder_units[1:]:
            new_layer = tf.layers.dense(new_layer, unit, tf.nn.elu)
        self.logits = tf.layers.dense(new_layer, self.n_in)
        self.decoder_op = tf.sigmoid(self.logits)
    # end method add_decoders


    def add_backward_path(self):
        self.reconstruct_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.X, logits=self.logits))
        self.latent_loss = 0.5 * tf.reduce_sum(tf.exp(self.gamma) + tf.square(self.mean) - 1 - self.gamma)
        self.loss = self.reconstruct_loss + self.latent_loss
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
    # end method add_backward_path


    def fit(self, X_train, val_data, n_epoch=10, batch_size=128, en_shuffle=True):
        self.sess.run(tf.global_variables_initializer()) # initialize all variables
        global_step = 0
        for epoch in range(n_epoch):
            X_train = sklearn.utils.shuffle(X_train)
            for local_step, X_batch in enumerate(self.gen_batch(X_train, batch_size)):
                _, loss, reconstruct_loss, latent_loss = self.sess.run([self.train_op, self.loss,
                                                                        self.reconstruct_loss, self.latent_loss],
                                                                       {self.X:X_batch})
                if local_step % 100 == 0:
                    print ("Epoch %d/%d | Step %d/%d | train loss: %.4f | reconstruct loss: %.4f | latent loss: %.4f"
                           %(epoch+1, n_epoch, local_step, len(X_train)//batch_size, loss, reconstruct_loss, latent_loss))
                global_step += 1
            
            val_loss_list = []
            for X_test_batch in self.gen_batch(val_data, batch_size):
                v_loss = self.sess.run(self.loss, {self.X:X_test_batch})
                val_loss_list.append(v_loss)
            val_loss = sum(val_loss_list) / len(val_loss_list)
            print ("Epoch %d/%d | train loss: %.4f | test loss: %.4f" %(epoch+1, n_epoch, loss, v_loss))
    # end method fit_transform


    def predict(self, X_test, batch_size=128):
        res = []
        for X_batch in self.gen_batch(X_test, batch_size):
            res.append(self.sess.run(self.decoder_op, {self.X: X_batch}))
        return np.vstack(res)
    # end method predict


    def generate(self, batch_size=128):
        return self.sess.run(self.decoder_op,
                            {self.encoder_op: np.random.randn(batch_size, self.encoder_units[-1])})
    # end method generate


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method gen_batch
# end class Autoencoder