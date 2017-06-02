import tensorflow as tf
import numpy as np


class CONV_GAN:
    def __init__(self, X_size, G_size=100, lr_G=1e-3, lr_D=1e-3):
        self.X_size = X_size # (height, width, channel)
        self.G_size = G_size
        self.lr_G = lr_G
        self.lr_D = lr_D
        self._depths = [64, 128, 256]
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.add_input_layer()
        with tf.variable_scope('G'):
            self.add_Generator()
        with tf.variable_scope('D'):
            self.add_Discriminator()
        self.add_backward_path()
    # end method build_graph


    def add_input_layer(self):
        self.G_in = tf.placeholder(tf.float32, [None, self.G_size]) # random data
        self.X_in = tf.placeholder(tf.float32, [None, self.X_size[0], self.X_size[1], self.X_size[2]]) # real data
        self.train_flag = tf.placeholder(tf.bool)


    def add_Generator(self):
        Y = tf.layers.dense(self.G_in, self._depths[2]*4*4, tf.nn.relu)
        Y = tf.reshape(Y, [-1, 4, 4, self._depths[2]])
        Y = tf.nn.relu(Y)

        Y = tf.layers.conv2d_transpose(Y, self._depths[1], (5,5), (2,2), 'SAME')
        Y = tf.contrib.layers.batch_norm(Y, is_training=self.train_flag)
        Y = tf.nn.relu(Y)

        Y = tf.layers.conv2d_transpose(Y, self._depths[0], (5,5), (2,2), 'SAME')
        Y = tf.contrib.layers.batch_norm(Y, is_training=self.train_flag)
        Y = tf.nn.relu(Y)

        Y = tf.layers.conv2d_transpose(Y, self.X_size[-1], (5,5), (2,2), 'SAME')
        Y = tf.contrib.layers.batch_norm(Y, is_training=self.train_flag) 
        Y = tf.nn.relu(Y)

        Y = tf.layers.conv2d(Y, self._depths[0], (5,5), (2,2), 'SAME')
        Y = tf.contrib.layers.batch_norm(Y, is_training=self.train_flag)
        Y = tf.nn.relu(Y)

        Y = tf.layers.conv2d(Y, self._depths[1], (5,5), (2,2), 'SAME')
        Y = tf.contrib.layers.batch_norm(Y, is_training=self.train_flag)
        Y = tf.nn.relu(Y)

        Y = tf.layers.conv2d(Y, self._depths[2], (5,5), (2,2), 'SAME')
        Y = tf.contrib.layers.batch_norm(Y, is_training=self.train_flag)
        Y = tf.nn.relu(Y)

        self.G_out = tf.reshape(Y, [-1, self.X_size[0] * self.X_size[1] * self._depths[2]])
    # end method add_Generator


    def add_Discriminator(self):
        Y = tf.layers.conv2d(self.X_in, self._depths[0], (5,5), (2,2), 'SAME')
        Y = tf.contrib.layers.batch_norm(Y, is_training=self.train_flag)
        Y = tf.nn.relu(Y)

        Y = tf.layers.conv2d(Y, self._depths[1], (5,5), (2,2), 'SAME')
        Y = tf.contrib.layers.batch_norm(Y, is_training=self.train_flag)
        Y = tf.nn.relu(Y)

        Y = tf.layers.conv2d(Y, self._depths[2], (5,5), (2,2), 'SAME')
        Y = tf.contrib.layers.batch_norm(Y, is_training=self.train_flag)
        Y = tf.nn.relu(Y)

        self.D_out = tf.reshape(Y, [-1, self.X_size[0] * self.X_size[1] * self._depths[2]])

        D_hidden = tf.layers.dense(self.D_out, 128, tf.nn.relu, name='hidden')
        self.X_true_prob = tf.layers.dense(D_hidden, 1, tf.nn.sigmoid, name='out')

        D_hidden = tf.layers.dense(self.G_out, 128, tf.nn.relu, name='hidden', reuse=True)
        self.G_true_prob = tf.layers.dense(D_hidden, 1, tf.nn.sigmoid, name='out', reuse=True)
    # end method add_Discriminator


    def add_backward_path(self):
        self.G_loss = - tf.reduce_mean(tf.log(self.G_true_prob))
        self.D_loss = - tf.reduce_mean(tf.log(self.X_true_prob) + tf.log(1 - self.G_true_prob))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.G_train = tf.train.AdamOptimizer(self.lr_G).minimize(self.G_loss,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G'))
            self.D_train = tf.train.AdamOptimizer(self.lr_D).minimize(self.D_loss,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D'))
    # end method add_backward_path


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method gen_batch

    
    def fit(self, X, batch_size=128, n_epoch=10):
        for epoch in range(n_epoch):
            for step, X_batch in enumerate(self.gen_batch(X_train, batch_size)):
                G_out, D_prob, D_loss, _, _ = sess.run([self.G_out, self.X_true_prob, self.D_loss, self.D_train, self.G_train],
                                                       {self.G_in: np.random.randn(batch_size, self.G_size),
                                                        self.X_in: X,
                                                        self.train_flag: True})
                print("Step %d | X true: %.2f | D loss: %.2f" % (step, D_prob, D_loss))
    # end method fit
# end class