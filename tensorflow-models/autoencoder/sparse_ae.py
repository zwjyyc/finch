import tensorflow as tf
import sklearn
import numpy as np
import math


class Autoencoder:
    def __init__(self, n_in, n_hidden=1000, sparsity_target=0.1, sparsity_weight=0.2, lr=0.01, sess=tf.Session()):
        self.sess = sess
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight
        self.lr = lr
        self.build_graph()
    # end constructor


    def kl_divergence(self, p, q):
        return p * tf.log(p / q) + (1 - p) * tf.log((1 - p) / (1 - q)) # Kullback Leibler divergence
    # end method kl_divergence


    def build_graph(self):
        self.add_forward_path()
        self.add_backward_path()
    # end method build_graph


    def add_forward_path(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.n_in])
        self.hidden = tf.layers.dense(self.X, self.n_hidden, tf.nn.sigmoid)
        self.logits = tf.layers.dense(self.hidden, self.n_in)
        self.decoder_op = tf.sigmoid(self.logits)
    # end method add_forward_path


    def add_backward_path(self):
        hidden_mean = tf.reduce_mean(self.hidden, axis=0) # batch mean
        self.sparsity_loss = tf.reduce_sum(self.kl_divergence(self.sparsity_target, hidden_mean))
        self.mse_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.X, logits=self.logits))
        self.loss = self.mse_loss + self.sparsity_weight * self.sparsity_loss
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    # end method add_backward_path


    def fit(self, X_train, val_data, n_epoch=10, batch_size=128, en_shuffle=True, keep_prob=0.8):
        self.sess.run(tf.global_variables_initializer()) # initialize all variables
        global_step = 0
        for epoch in range(n_epoch):
            X_train = sklearn.utils.shuffle(X_train)
            for local_step, X_batch in enumerate(self.gen_batch(X_train, batch_size)):
                _, loss, mse, sparsity = self.sess.run([self.train_op, self.loss, self.mse_loss, self.sparsity_loss],
                                                       {self.X:X_batch})
                if local_step % 100 == 0:
                    print ("Epoch %d/%d | Step %d/%d | train loss: %.4f | mse loss: %.4f | sparsity loss: %.4f"
                           %(epoch+1, n_epoch, local_step, len(X_train)//batch_size, loss, mse, sparsity))
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


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method gen_batch
# end class Autoencoder