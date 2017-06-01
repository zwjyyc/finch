import tensorflow as tf
import math
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import string
import re
import collections
import sys


class RNNTextGen:
    def __init__(self, text, seq_len=50, embedding_dims=128, cell_size=128, n_layer=2, grad_clip=5,
                 useless_words=None, sess=tf.Session()):
        """
        Parameters:
        -----------
        sess: object
            tf.Session() object
        text: string
            corpus in one long string, usually obtained by file.read()
        seq_len: int
            Sequence length
        embedding_dims: int
            length of embedding vector for each word
        cell_size: int
            Number of units in the rnn cell, default to 128
        n_layers: int
            Number of layers of stacked rnn cells
        useless_words: list of characters
            all the useless_words which will be removed from text, usually punctuations
        """
        self.sess = sess
        self.text = text
        self.seq_len = seq_len
        self.embedding_dims = embedding_dims
        self.cell_size = cell_size
        self.n_layer = n_layer
        self.useless_words = useless_words
        self.grad_clip = grad_clip
        self._cursor = None
        self.preprocessing()
        self.build_graph()
    # end constructor


    def build_graph(self):
        with tf.variable_scope('main_model'):
            self.add_input_layer()       
            self.add_word_embedding()
            self.add_lstm_cells()
            self.add_dynamic_rnn()
            self.add_output_layer()
            self.add_backward_path()
        with tf.variable_scope('main_model', reuse=True):
            self.add_sample_model()
    # end method build_graph


    def add_input_layer(self):
        self.X = tf.placeholder(tf.int32, [None, self.seq_len])
        self.Y = tf.placeholder(tf.int32, [None, self.seq_len])
        self.batch_size = tf.placeholder(tf.int32)
        self.lr = tf.placeholder(tf.float32) 
        self._cursor = self.X
    # end method add_input_layer


    def add_word_embedding(self):
        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_dims)
        E = tf.get_variable('E', [self.vocab_size, self.embedding_dims], tf.float32, tf.random_normal_initializer())
        self._cursor = tf.nn.embedding_lookup(E, self._cursor)
    # end method add_word_embedding


    def add_lstm_cells(self):
        if tf.__version__[0] == 1:
            def cell():
                cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size)
                return cell
            self.cells = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(self.n_layer)])
        else:
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size)
            self.cells = tf.contrib.rnn.MultiRNNCell([cell * self.n_layer])
    # end method add_rnn_cells


    def add_dynamic_rnn(self):
        self.init_state = self.cells.zero_state(self.batch_size, tf.float32)
        self._cursor, final_state = tf.nn.dynamic_rnn(self.cells, self._cursor,
                                                      initial_state=self.init_state, time_major=False)   
    # end method add_dynamic_rnn


    def add_output_layer(self):
        reshaped = tf.reshape(self._cursor, [-1, self.cell_size])
        self.logits = tf.layers.dense(reshaped, self.vocab_size, name='output')
    # end method add_output_layer


    def add_backward_path(self):
        losses = tf.contrib.seq2seq.sequence_loss(
            logits = tf.reshape(self.logits, [self.batch_size, self.seq_len, self.vocab_size]),
            targets = self.Y,
            weights = tf.ones([self.batch_size, self.seq_len]),
            average_across_timesteps = True,
            average_across_batch = True,
        )
        self.loss = tf.reduce_sum(losses)
        # gradient clipping
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
    # end method add_backward_path


    def add_sample_model(self):
        self.X_ = tf.placeholder(tf.int32, [None, 1])
        self.init_state_ = self.cells.zero_state(self.batch_size, tf.float32)
        X = tf.nn.embedding_lookup(tf.get_variable('E'), self.X_)
        Y, self.final_state_ = tf.nn.dynamic_rnn(self.cells, X, initial_state=self.init_state_, time_major=False)
        Y = tf.reshape(Y, [-1, self.cell_size])
        Y = tf.layers.dense(Y, self.vocab_size, name='output', reuse=True)
        self.softmax_out_ = tf.nn.softmax(Y)
    # end add_sample_model


    def adjust_lr(self, current_step, total_steps):
        max_lr = 0.003
        min_lr = 0.0001
        decay_rate = math.log(min_lr/max_lr) / (-total_steps)
        lr = max_lr * math.exp(-decay_rate * current_step)
        return lr
    # end method adjust_lr


    def preprocessing(self):
        text = self.text
        text = text.replace('\n', ' ')
        if self.useless_words is None:
            self.useless_words = string.punctuation
        if sys.version[0] == 3:
            table = str.maketrans( {useless: ' ' for useless in self.useless_words} )
            text = text.translate(table)
        else:
            text = re.sub(r'[{}]'.format(''.join(self.useless_words)), ' ', text)
        text = re.sub('\s+', ' ', text ).strip().lower()
        
        chars = list(set(text))
        self.char2idx = {c: i for i, c in enumerate(chars)}
        self.idx2char = {i: c for i, c in enumerate(chars)}
        assert len(self.idx2char) == len(self.char2idx), "len(idx2char) != len(char2idx)"
        self.vocab_size = len(self.idx2char)
        print('Vocabulary size:', self.vocab_size)

        self.indices = [self.char2idx[char] for char in list(text)]
    # end method text_preprocessing


    def fit(self, prime_texts, text_iter_step=10, n_gen=500, n_epoch=20, batch_size=128,
            en_exp_decay=True, en_shuffle=True):
        window = self.seq_len + 1
        X = np.array([self.indices[i:i+window] for i in range(0, len(self.indices)-window, text_iter_step)])
        Y = np.roll(X, -1, axis=1)
        X = X[:, :-1]
        Y = Y[:, :-1]
        print('X shape:', X.shape, '|', 'Y shape:', Y.shape)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        global_step = 0
        n_batch = len(X_train) / batch_size
        self.sess.run(tf.global_variables_initializer()) # initialize all variables
        
        for epoch in range(n_epoch):
            next_state = self.sess.run(self.init_state, feed_dict={self.batch_size:batch_size})
            local_step = 1
            if en_shuffle:
                X_train, Y_train = shuffle(X_train, Y_train)
            for X_train_batch, Y_train_batch in zip(self.gen_batch(X_train, batch_size),
                                                    self.gen_batch(Y_train, batch_size)):
                lr = self.adjust_lr(global_step, int(n_epoch * n_batch)) if en_exp_decay else 0.001
                _, train_loss = self.sess.run([self.train_op, self.loss],
                                              {self.X:X_train_batch, self.Y:Y_train_batch,
                                               self.batch_size:len(X_train_batch), self.lr:lr})
                if local_step % 10 == 0:
                    print ('Epoch %d/%d | Batch %d/%d | train loss: %.4f | lr: %.4f'
                            % (epoch+1, n_epoch, local_step, n_batch, train_loss, lr))
                if local_step % 100 == 0:
                    test_losses = []
                    for X_test_batch, Y_test_batch in zip(self.gen_batch(X_test, batch_size),
                                                          self.gen_batch(Y_test, batch_size)):
                        test_loss = self.sess.run(self.loss, {self.X:X_test_batch, self.Y:Y_test_batch,
                                                              self.batch_size:len(X_test_batch)})
                        test_losses.append(test_loss)
                    avg_test_loss = sum(test_losses) / len(test_losses)
                    print ('Epoch %d/%d | Batch %d/%d | train loss: %.4f | test loss: %.4f'
                            % (epoch+1, n_epoch, local_step, n_batch, train_loss, avg_test_loss))
                    for prime_text in prime_texts:
                            print(self.predict(prime_text, n_gen)+'\n')
                local_step += 1
                global_step += 1
            
        return log
    # end method fit


    def predict(self, prime_text, n_gen):
        # warming up
        next_state = self.sess.run(self.init_state_, {self.batch_size:1})
        char_list = list(prime_text)
        for char in char_list[:-1]:
            x = np.atleast_2d(self.char2idx[char]) 
            next_state = self.sess.run(self.final_state_, {self.X_:x, self.init_state_:next_state})
        # end warming up

        out_sentence = 'IN: ' + prime_text + '\nOUT: ' + prime_text
        char = char_list[-1]
        for _ in range(n_gen):
            x = np.atleast_2d(self.char2idx[char])
            softmax_out, next_state = self.sess.run([self.softmax_out_, self.final_state_],
                                                    {self.X_:x, self.init_state_:next_state})
            probas = softmax_out[0].astype('float64')
            probas = probas / np.sum(probas)
            actions = np.random.multinomial(1, probas, 1)
            idx = np.argmax(actions)
            char = self.idx2char[idx]
            out_sentence = out_sentence + char
        return out_sentence
    # end method sample


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method gen_batch
# end class
