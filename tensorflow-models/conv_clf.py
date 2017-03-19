import tensorflow as tf
import numpy as np
import math


class ConvClassifier:
    def __init__(self, width, height, n_out=2):
        self.width = width
        self.height = height
        self.n_out = n_out

        self.build_graph()
    # end constructor


    def build_graph(self):
        self.X = tf.placeholder(tf.float32, [None, self.width, self.height, 1])
        self.y = tf.placeholder(tf.float32, [None, self.n_out])

        self.W = {
            'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])), # 5x5 conv, 1 input, 32 outputs
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])), # 5x5 conv, 32 inputs, 64 outputs
            # fully connected
            'wd1': tf.Variable(tf.random_normal([int(self.width/4) * int(self.height/4) * 64, 1024])),
            'out': tf.Variable(tf.random_normal([1024, self.n_out])) # class prediction
        }
        self.b = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([self.n_out]))
        }

        self.lr = tf.placeholder(tf.float32)

        self.pred = self.conv(self.X, self.W, self.b)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
        self.train = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.acc = tf.reduce_mean( tf.cast( tf.equal( tf.argmax(self.pred, 1), tf.argmax(self.y, 1) ), tf.float32 ) )

        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
    # end method build_graph


    def conv(self, X, W, b):
        conv1 = self.conv2d(X, W['wc1'], b['bc1'])       # convolution layer
        conv1 = self.maxpool2d(conv1, k=2)               # max pooling (down-sampling)

        conv2 = self.conv2d(conv1, W['wc2'], b['bc2'])   # convolution layer
        conv2 = self.maxpool2d(conv2, k=2)               # max pooling (down-sampling)

        # fully connected layer, reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, W['wd1'].get_shape().as_list()[0]])
        fc1 = tf.nn.bias_add(tf.matmul(fc1, W['wd1']),b['bd1'])
        fc1 = tf.nn.relu(tf.contrib.layers.batch_norm(fc1))

        # output, class prediction
        out = tf.nn.bias_add(tf.matmul(fc1, W['out']), b['out'])
        return out
    # end method conv


    def conv2d(self, X, W, b, strides=1):
        conv = tf.nn.conv2d(X, W, strides=[1, strides, strides, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, b)
        return tf.nn.relu(tf.contrib.layers.batch_norm(conv))


    def maxpool2d(self, X, k=2):
        return tf.nn.max_pool(X, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


    def fit(self, X, y, validation_data, n_epoch=10, batch_size=32):
        print("Train %d samples | Test %d samples" % (len(X), len(validation_data[0])))
        log = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
        max_lr = 0.003
        min_lr = 0.0001
        decay_rate = math.log(min_lr/max_lr) / (-n_epoch*len(X)/batch_size)

        self.sess.run(self.init) # initialize all variables

        global_step = 0
        for epoch in range(n_epoch):
            # batch training
            for X_batch, y_batch in zip(self.gen_batch(X, batch_size), self.gen_batch(y, batch_size)):
                self.sess.run(self.train, feed_dict={self.X: X_batch, self.y: y_batch,
                                                     self.lr: max_lr*math.exp(-decay_rate*global_step)})
                global_step += 1
            # compute training loss and acc
            loss, acc = self.sess.run([self.loss, self.acc], feed_dict={self.X: X_batch, self.y: y_batch})
            # compute validation loss and acc
            val_loss_list, val_acc_list = [], []
            for X_test_batch, y_test_batch in zip(self.gen_batch(validation_data[0], batch_size),
                                                  self.gen_batch(validation_data[1], batch_size)):
                val_loss_list.append(self.sess.run(self.loss, feed_dict={self.X: X_test_batch,
                                                                         self.y: y_test_batch}))
                val_acc_list.append(self.sess.run(self.acc, feed_dict={self.X: X_test_batch,
                                                                       self.y: y_test_batch}))
            val_loss, val_acc = sum(val_loss_list)/len(val_loss_list), sum(val_acc_list)/len(val_acc_list)

            # append to log
            log['loss'].append(loss)
            log['acc'].append(acc)
            log['val_loss'].append(val_loss)
            log['val_acc'].append(val_acc)

            # verbose
            print ("%d / %d: train_loss: %.4f train_acc: %.4f | test_loss: %.4f test_acc: %.4f"
                   % (epoch+1, n_epoch, loss, acc, val_loss, val_acc))
            
        return log
    # end method fit


    def predict(self, X_test, batch_size=32):
        batch_pred_list = []
        for X_test_batch in self.gen_batch(X_test, batch_size):
            batch_pred = self.sess.run(self.pred, feed_dict={self.X: X_test_batch})
            batch_pred_list.append(batch_pred)
        return np.concatenate(batch_pred_list)
    # end method predict


    def close(self):
        self.sess.close()
        tf.reset_default_graph()
    # end method close


    def gen_batch(self, arr, batch_size):
        if len(arr) % batch_size != 0:
            new_len = len(arr) - len(arr) % batch_size
            for i in range(0, new_len, batch_size):
                yield arr[i : i + batch_size]
        else:
            for i in range(0, len(arr), batch_size):
                yield arr[i : i + batch_size]
    # end method gen_batch
# end class LinearSVMClassifier
