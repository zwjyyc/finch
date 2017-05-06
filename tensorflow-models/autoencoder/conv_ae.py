import tensorflow as tf
import numpy as np
import math


class ConvAE:
    def __init__(self, sess, img_size, img_ch, kernel_size=(5,5), pool_size=2):
        self.img_size = img_size
        self.img_ch = img_ch
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.sess = sess
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.batch_size = tf.placeholder(tf.int32)
        self.X = tf.placeholder(tf.float32, [None, self.img_size[0], self.img_size[1], self.img_ch])
        strides=1
        W = tf.Variable(tf.truncated_normal([self.kernel_size[0], self.kernel_size[1], self.img_ch, 32],
                        stddev=0.1))
        conv = tf.nn.conv2d(self.X, W, strides=[1,strides,strides,1], padding='SAME')

        self.decoder_op = tf.nn.conv2d_transpose(conv, W,
                                                 [self.batch_size,self.img_size[0],self.img_size[1],self.img_ch],
                                                 [1,strides,strides,1], 'SAME')
        self.add_backward_path()
    # end method build_graph


    def add_backward_path(self):
        self.loss = tf.reduce_mean(tf.square(self.X - self.decoder_op))
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
    # end method add_backward_path


    def fit_transform(self, X_train, n_epoch=10, batch_size=32):
        self.sess.run(tf.global_variables_initializer()) # initialize all variables
        global_step = 0
        for epoch in range(n_epoch):
            # batch training
            for local_step, X_batch in enumerate(self.gen_batch(X_train, batch_size)):
                _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.X:X_batch,
                                                                               self.batch_size:len(X_batch)})
                if global_step == 0:
                    print("Initial loss: ", loss)
                if (local_step + 1) % 100 == 0:
                    print ("Epoch %d/%d | Step %d/%d | train loss: %.4f"
                           %(epoch+1, n_epoch, local_step+1, int(len(X_train)/batch_size), loss))
                global_step += 1

        res = []
        for X_batch in self.gen_batch(X_train, batch_size):
            res.append(self.sess.run(self.decoder_op, feed_dict={self.X: X_batch,
                                                                 self.batch_size:len(X_batch)}))
        return np.concatenate(res)
    # end method fit_transform


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method gen_batch
# end class Autoencoder
