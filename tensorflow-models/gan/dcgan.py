import tensorflow as tf


class Conv_GAN:
    def __init__(self, G_size, batch_size):
        self.G_size = G_size
        self.batch_size = batch_size
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
        self.G_in = tf.random_uniform([self.batch_size, self.G_size], minval=-1.0, maxval=1.0)
        self.X_in = tf.placeholder(tf.float32, [self.batch_size, 28, 28, 1])
        self.train_flag = tf.placeholder(tf.bool)
    # end method input_layer


    def add_Generator(self):
        def deconv(X):
            # 100 -> (7, 7, 64) ->  (14, 14, 32) -> (28, 28, 1)
            X = tf.layers.dense(X, 7 * 7 * 64)
            X = tf.reshape(X, [-1, 7, 7, 64])

            Y = tf.layers.conv2d_transpose(X, 32, [5, 5], strides=(2, 2), padding='SAME')
            Y = tf.layers.batch_normalization(Y, training=self.train_flag)
            Y = tf.nn.relu(Y)

            Y = tf.layers.conv2d_transpose(Y, 1, [5, 5], strides=(2, 2), padding='SAME')
            return Y
        
        self.G_out = deconv(self.G_in)
    # end method add_Generator


    def add_Discriminator(self):
        def lrelu(X, leak=0.2):
            return tf.maximum(X, X * leak)
        
        def conv(X, reuse=False):
            # (28, 28, 1) -> (14, 14, 32) -> (7, 7, 64) -> 1
            Y = tf.layers.conv2d(X, 32, [5, 5], strides=(2, 2), padding='SAME', name='conv1', reuse=reuse)
            Y = tf.layers.batch_normalization(Y, training=self.train_flag, name='bn1', reuse=reuse)
            Y = lrelu(Y)

            Y = tf.layers.conv2d(Y, 64, [5, 5], strides=(2, 2), padding='SAME', name='conv2', reuse=reuse)
            Y = tf.layers.batch_normalization(Y, training=self.train_flag, name='bn2', reuse=reuse)
            Y = lrelu(Y)

            flat = tf.reshape(Y, [-1, 7 * 7 * 64])
            output = tf.layers.dense(flat, 1, name='out', reuse=reuse)
            return output
        
        self.G_logits = conv(self.G_out)
        self.X_logits = conv(self.X_in, reuse=True)
        self.G_prob = tf.nn.sigmoid(self.G_logits)
        self.X_prob = tf.nn.sigmoid(self.X_logits)
    # end method add_Discriminator


    def add_backward_path(self):
        ones = tf.ones([self.batch_size, 1], tf.float32)
        zeros = tf.zeros([self.batch_size, 1], tf.float32)

        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ones, logits=self.G_logits))
        D_loss_X = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ones, logits=self.X_logits))
        D_loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=zeros, logits=self.G_logits))
        self.D_loss = D_loss_X + D_loss_G

        G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='G')
        with tf.control_dependencies(G_update_ops):
            self.G_train = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.G_loss,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G'))

        D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='D')
        with tf.control_dependencies(D_update_ops):
            self.D_train = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.D_loss,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D'))

        self.mse = tf.reduce_mean(tf.squared_difference(self.G_out, self.X_in))
    # end method add_backward_path
# end class