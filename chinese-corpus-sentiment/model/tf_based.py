import tensorflow as tf
import os
import numpy as np


class RNNClassifier:
    def __init__(self, n_step, n_in, n_hidden_units, n_out):
        # set random seed to fix the weights and biases		
        tf.set_random_seed(21)

        # neural network parameters
        n_step = n_step
        n_in = n_in
        n_hidden_units = n_hidden_units
        n_out = n_out

        # tf graph inputs
        self.X = tf.placeholder(tf.float32, [None, n_step, n_in])
        self.y = tf.placeholder(tf.int64)

        # define weights and biases
        self.W = tf.Variable(tf.random_normal([n_hidden_units, n_out]))
        self.b = tf.Variable(tf.random_normal([n_out]))

        # start rnn calculation
        X = tf.transpose(self.X, [1, 0, 2])
        X = tf.reshape(X, [-1, n_in])
        X = tf.split(X, n_step, 0)

        lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden_units, state_is_tuple=True)
        lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=1.0, output_keep_prob=0.5)
        outputs, states = tf.contrib.rnn.static_rnn(cell=lstm_cell, inputs=X, dtype=tf.float32)
        self.pred = tf.add(tf.matmul(outputs[-1], self.W), self.b)
        # end rnn calculation

        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
        self.init_lr = 1e-4
        self.lr = tf.placeholder(tf.float32)
        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

        # evaluate model
        self.is_correct = tf.equal(tf.argmax(self.pred, 1), self.y)
        self.acc = tf.reduce_mean(tf.cast(self.is_correct, tf.float32))

        # initialization
        self.sess = tf.Session()
        self.init_op = tf.global_variables_initializer()


    def fit(self, X_train, y_train, validation_data, nb_epoch=10, batch_size=32):
        print "Train %d samples | Test %d samples" % (len(X_train), len(validation_data[0]))
        log = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
        nb_batch = int( len(X_train) / batch_size )

        self.sess.run(self.init_op)
        epoch = 1
        X_train_batch_list = np.array_split(X_train, nb_batch)
        y_train_batch_list = np.array_split(y_train, nb_batch)
        for _ in range(nb_epoch):
            for X_train_batch, y_train_batch in zip(X_train_batch_list, y_train_batch_list):
                self.sess.run(self.optimize, feed_dict={self.X: X_train_batch, self.y: y_train_batch, self.lr: self.lr_fn(log)})
            loss, acc = self.sess.run([self.cost, self.acc],
                                      feed_dict={self.X: X_train_batch, self.y: y_train_batch})
            val_loss, val_acc = self.sess.run([self.cost, self.acc],
                                              feed_dict={self.X: validation_data[0], self.y: validation_data[1]})
            print "%d / %d: train_loss: %.5f train_acc: %.2f | test_loss: %.5f test_acc: %.2f" % (epoch, nb_epoch, loss, acc, val_loss, val_acc)
            log['loss'].append(loss)
            log['acc'].append(acc)
            log['val_loss'].append(val_loss)
            log['val_acc'].append(val_acc)
            epoch += 1

        return log


    def predict(self, X_test):
        y_test_pred = self.sess.run(self.pred, feed_dict={self.X: X_test})
        return np.argmax(y_test_pred, axis=1)


    def lr_fn(self, log):
        if len(log['val_loss']) < 2:
            return self.init_lr
        else:
            if log['val_loss'][-1] > log['val_loss'][-2]:
                return self.init_lr / 2
            else:
                return self.init_lr
    # end method lr_fn
# end class RNNClassifier


class RNNClassifier_:
    def __init__(self, n_step, n_in, n_hidden_units, n_out):
        # set random seed to fix the weights and biases		
        tf.set_random_seed(21)

        # neural network parameters
        n_step = n_step
        n_in = n_in
        n_hidden_units = n_hidden_units
        n_out = n_out

        # tf graph inputs
        self.X = tf.placeholder(tf.float32, [None, n_step, n_in])
        self.y = tf.placeholder(tf.int64)

        # define weights and biases
        self.W = tf.Variable(tf.random_normal([n_hidden_units, n_out]))
        self.b = tf.Variable(tf.random_normal([n_out]))

        # start rnn calculation
        X = tf.transpose(self.X, [1, 0, 2])
        X = tf.reshape(X, [-1, n_in])
        X = tf.split(X, n_step, 0)

        lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden_units, state_is_tuple=True)
        lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=1.0, output_keep_prob=0.5)
        outputs, states = tf.contrib.rnn.static_rnn(cell=lstm_cell, inputs=X, dtype=tf.float32)
        self.pred = tf.add(tf.matmul(outputs[-1], self.W), self.b)
        # end rnn calculation

        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
        self.init_lr = 1e-4
        self.lr = tf.placeholder(tf.float32)
        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

        # evaluate model
        self.is_correct = tf.equal(tf.argmax(self.pred, 1), self.y)
        self.acc = tf.reduce_mean(tf.cast(self.is_correct, tf.float32))

        # initialization
        self.init_op = tf.global_variables_initializer()

        # save and restore all the variables
        self.saver = tf.train.Saver()
        self.model_path = os.path.join(os.path.dirname(os.getcwd()), 'model', 'save_model', 'rnn.ckpt')


    def fit(self, X_train, y_train, validation_data, nb_epoch=10, batch_size=32):
        print "Train %d samples | Test %d samples" % (len(X_train), len(validation_data[0]))
        log = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
        nb_batch = int( len(X_train) / batch_size )
        with tf.Session() as sess:
            sess.run(self.init_op)
            epoch = 1
            X_train_batch_list = np.array_split(X_train, nb_batch)
            y_train_batch_list = np.array_split(y_train, nb_batch)
            for _ in range(nb_epoch):
                for X_train_batch, y_train_batch in zip(X_train_batch_list, y_train_batch_list):
                    sess.run(self.optimize, feed_dict={self.X: X_train_batch, self.y: y_train_batch, self.lr: self.lr_fn(log)})
                loss, acc = sess.run([self.cost, self.acc],
                                     feed_dict={self.X: X_train_batch, self.y: y_train_batch})
                val_loss, val_acc = sess.run([self.cost, self.acc],
                                             feed_dict={self.X: validation_data[0], self.y: validation_data[1]})
                print "%d / %d: train_loss: %.5f train_acc: %.2f | test_loss: %.5f test_acc: %.2f" % (epoch, nb_epoch, loss, acc, val_loss, val_acc)
                log['loss'].append(loss)
                log['acc'].append(acc)
                log['val_loss'].append(val_loss)
                log['val_acc'].append(val_acc)
                epoch += 1
            save_path = self.saver.save(sess, self.model_path) # save model weights to disk
            print "Model saved in: %s" % save_path
        return log


    def predict(self, X_test):
        with tf.Session() as sess:
            sess.run(self.init_op)
            self.saver.restore(sess, self.model_path) # restore model weights from disk
            y_test_pred = sess.run(self.pred, feed_dict={self.X: X_test})
        return np.argmax(y_test_pred, axis=1)


    def lr_fn(self, log):
        if len(log['val_loss']) < 2:
            return self.init_lr
        else:
            if log['val_loss'][-1] > log['val_loss'][-2]:
                return self.init_lr / 2
            else:
                return self.init_lr

