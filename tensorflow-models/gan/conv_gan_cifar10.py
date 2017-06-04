import tensorflow as tf
import numpy as np


class ConvGAN:
    def __init__(self, X_size, G_size=100, lr_G=2e-4, lr_D=2e-4, sess=tf.Session() ):
        self.G_size = G_size
        self.X_size = X_size # (height, width, channel)
        self.lr_G = lr_G
        self.lr_D = lr_D
        self.sess = sess
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
        # 100 -> (4, 4, 512) -> (8, 8, 256) -> (16, 16, 128) -> (32, 32, 3)
        # 100 -> (4, 4, 512)
        X = tf.layers.dense(self.G_in, 4 * 4 * 512)
        X = tf.reshape(X, [-1, 4, 4, 512])
        X = tf.nn.relu(tf.contrib.layers.batch_norm(X, is_training=self.train_flag))
        # (4, 4, 512) -> (8, 8, 256)
        Y = tf.layers.conv2d_transpose(X, 256, [5, 5], strides=(2, 2), padding='SAME')
        Y = tf.nn.relu(tf.contrib.layers.batch_norm(Y, is_training=self.train_flag))
        # (8, 8, 256) -> (16, 16, 128)
        Y = tf.layers.conv2d_transpose(Y, 128, [5, 5], strides=(2, 2), padding='SAME')
        Y = tf.nn.relu(tf.contrib.layers.batch_norm(Y, is_training=self.train_flag))
        # (16, 16, 128) -> (32, 32, 3)
        Y = tf.layers.conv2d_transpose(Y, 3, [5, 5], strides=(2, 2), padding='SAME')
        self.G_out = tf.nn.tanh(Y)
    # end method add_Generator


    def add_Discriminator(self):
        def leaky_relu(x, leak=0.2):
            return tf.maximum(x, x * leak)
        
        def compute(X, reuse=False):
            # (32, 32, 3) -> (16, 16, 64) -> (8, 8, 128) -> (4, 4, 256) -> 1
            # (32, 32, 3) -> (16, 16, 64)
            Y = tf.layers.conv2d(X, 64, [5, 5], strides=(2, 2), padding='SAME', name='conv1', reuse=reuse)
            Y = leaky_relu(tf.contrib.layers.batch_norm(Y, is_training=self.train_flag, scope='bn1', reuse=reuse))
            # (16, 16, 64) -> (8, 8, 128)
            Y = tf.layers.conv2d(Y, 128, [5, 5], strides=(2, 2), padding='SAME', name='conv2', reuse=reuse)
            Y = leaky_relu(tf.contrib.layers.batch_norm(Y, is_training=self.train_flag, scope='bn2', reuse=reuse))
            # (8, 8, 128) -> (4, 4, 256)
            Y = tf.layers.conv2d(Y, 256, [5, 5], strides=(2, 2), padding='SAME', name='conv3', reuse=reuse)
            Y = leaky_relu(tf.contrib.layers.batch_norm(Y, is_training=self.train_flag, scope='bn3', reuse=reuse))
            # (4, 4, 256) -> 1
            fully_connected = tf.reshape(Y, [-1, 4 * 4 * 256])
            return tf.layers.dense(fully_connected, 1, name='out', reuse=reuse)
        
        self.G_logits = compute(self.G_out)
        self.X_logits = compute(self.X_in, reuse=True)
        self.X_true_prob = tf.nn.sigmoid(self.X_logits)
    # end method add_Discriminator


    def add_backward_path(self):
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(self.G_logits), logits=self.G_logits))

        D_loss_X = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(self.X_logits), logits=self.X_logits))

        D_loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(self.G_logits), logits=self.G_logits))

        self.D_loss = D_loss_X + D_loss_G

        G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='G')
        with tf.control_dependencies(G_update_ops):
            self.G_train = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.G_loss,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G'))

        D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='D')
        with tf.control_dependencies(D_update_ops):
            self.D_train = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.D_loss,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D'))
    # end method add_backward_path


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method gen_batch
# end class