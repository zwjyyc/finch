import tensorflow as tf
import numpy as np
import math
from sklearn.utils import shuffle


class BiRNN:
    def __init__(self, vocab_size, n_out, embedding_dims=128, cell_size=128, n_layer=1, sess=tf.Session()):
        """
        Parameters:
        -----------
        vocab_size: int
            Vocabulary size
        cell_size: int
            Number of units in the rnn cell
        n_out: int
            Output dimensions
        sess: object
            tf.Session() object
        stateful: boolean
            If true, the final state for each batch will be used as the initial state for the next batch 
        """
        self.vocab_size = vocab_size
        self.embedding_dims = embedding_dims
        self.cell_size = cell_size
        self.n_layer = n_layer
        self.n_out = n_out
        self.sess = sess
        self._pointer = None
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.add_input_layer()
        self.add_word_embedding_layer()
        self.add_bidirectional_dynamic_rnn()
        self.add_output_layer()
        self.add_backward_path()
    # end method build_graph


    def add_input_layer(self):
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int64, [None, None])
        self.batch_size = tf.placeholder(tf.int32)
        self.keep_prob = tf.placeholder(tf.float32)
        self.lr = tf.placeholder(tf.float32)
        self._pointer = self.X
    # end method add_input_layer


    def add_word_embedding_layer(self):
        embedding = tf.get_variable('encoder', [self.vocab_size, self.embedding_dims], tf.float32,
                                     tf.random_uniform_initializer(-1.0, 1.0))
        embedded = tf.nn.embedding_lookup(embedding, self._pointer)
        self._pointer = tf.nn.dropout(embedded, self.keep_prob)
    # end method add_word_embedding_layer


    def lstm_cell(self):
        return tf.nn.rnn_cell.LSTMCell(self.cell_size, initializer=tf.orthogonal_initializer())
    # end method lstm_cell


    def add_bidirectional_dynamic_rnn(self):
        birnn_out = self._pointer
        for n in range(self.n_layer):
            (out_fw, out_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = self.lstm_cell(), cell_bw = self.lstm_cell(),
                inputs = birnn_out,
                dtype = tf.float32,
                scope = 'birnn%d'%n)
            birnn_out = tf.concat((out_fw, out_bw), 2)
        self._pointer = birnn_out
    # end method add_dynamic_rnn


    def add_output_layer(self):
        self.logits = tf.layers.dense(tf.reshape(self._pointer, [-1, 2*self.cell_size]), self.n_out)
    # end method add_output_layer


    def add_backward_path(self):
        self.loss = tf.contrib.seq2seq.sequence_loss(
            logits = tf.reshape(self.logits, [self.batch_size, -1, self.n_out]),
            targets = self.Y,
            weights = tf.ones_like(self.X, tf.float32),
            average_across_timesteps = True,
            average_across_batch = True,
        )
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1),
                                                   tf.reshape(self.Y, [-1])), tf.float32))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    # end method add_backward_path


    def fit(self, X, Y, val_data=None, n_epoch=10, batch_size=128, en_exp_decay=True, en_shuffle=True,
            keep_prob=1.0):
        if val_data is None:
            print("Train %d samples" % len(X) )
        else:
            print("Train %d samples | Test %d samples" % (len(X), len(val_data[0])))
        log = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
        global_step = 0

        self.sess.run(tf.global_variables_initializer()) # initialize all variables
        for epoch in range(n_epoch): # batch training
            if en_shuffle:
                X, Y = shuffle(X, Y)
                print("Data Shuffled")
            for local_step, (X_batch, Y_batch) in enumerate(zip(self.gen_batch(X, batch_size),
                                                                self.gen_batch(Y, batch_size))):
                lr = self.decrease_lr(en_exp_decay, global_step, n_epoch, len(X), batch_size)           
                _, loss, acc = self.sess.run([self.train_op, self.loss, self.acc],
                                             {self.X: X_batch, self.Y: Y_batch,
                                              self.batch_size: len(X_batch),
                                              self.lr: lr,
                                              self.keep_prob: keep_prob})
                global_step += 1
                if local_step % 50 == 0:
                    print ('Epoch %d/%d | Step %d/%d | train_loss: %.4f | train_acc: %.4f | lr: %.4f'
                           %(epoch+1, n_epoch, local_step, int(len(X)/batch_size), loss, acc, lr))

            if val_data is not None: # go through testing data, average validation loss and ac 
                val_loss_list, val_acc_list = [], []
                for X_test_batch, Y_test_batch in zip(self.gen_batch(val_data[0], batch_size),
                                                      self.gen_batch(val_data[1], batch_size)):
                    v_loss, v_acc = self.sess.run([self.loss, self.acc],
                                                  {self.X: X_test_batch, self.Y: Y_test_batch,
                                                   self.batch_size: len(X_test_batch),
                                                   self.keep_prob: 1.0})
                    val_loss_list.append(v_loss)
                    val_acc_list.append(v_acc)
                val_loss, val_acc = self.list_avg(val_loss_list), self.list_avg(val_acc_list)

            # append to log
            log['loss'].append(loss)
            log['acc'].append(acc)
            if val_data is not None:
                log['val_loss'].append(val_loss)
                log['val_acc'].append(val_acc)

            # verbose
            if val_data is None:
                print ("Epoch %d/%d | train_loss: %.4f | train_acc: %.4f |" % (epoch+1, n_epoch, loss, acc),
                       "lr: %.4f" % (lr) )
            else:
                print ("Epoch %d/%d | train_loss: %.4f | train_acc: %.4f |" % (epoch+1, n_epoch, loss, acc),
                       "test_loss: %.4f | test_acc: %.4f |" % (val_loss, val_acc), "lr: %.4f" % (lr) )
        # end "for epoch in range(n_epoch)"

        return log
    # end method fit


    def predict(self, X_test, batch_size=128):
        batch_pred_list = []
        for X_test_batch in self.gen_batch(X_test, batch_size):
            batch_pred = self.sess.run(self.logits,
                                      {self.X: X_test_batch,
                                       self.batch_size: len(X_test_batch),
                                       self.keep_prob: 1.0})
            batch_pred_list.append(batch_pred)
        return np.argmax(np.vstack(batch_pred_list), 1)
    # end method predict


    def infer(self, xs):
        logits = self.sess.run(self.logits, {self.X: np.atleast_2d(xs),
                                             self.batch_size: 1,
                                             self.keep_prob: 1.0})
        return np.argmax(logits, 1)
    # end method infer


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method gen_batch


    def decrease_lr(self, en_exp_decay, global_step, n_epoch, len_X, batch_size):
        if en_exp_decay:
            max_lr = 0.005
            min_lr = 0.0005
            decay_rate = math.log(min_lr/max_lr) / (-n_epoch*len_X/batch_size)
            lr = max_lr*math.exp(-decay_rate*global_step)
        else:
            lr = 0.001
        return lr
    # end method adjust_lr


    def list_avg(self, l):
        return sum(l) / len(l)
    # end method list_avg
# end class