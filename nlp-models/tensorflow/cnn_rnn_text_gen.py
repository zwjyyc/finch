import tensorflow as tf
import math
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import string
import re
import collections
import sys


class ConvRNNTextGen:
    def __init__(self, text, seq_len=50, embedding_dims=15,
                 cell_size=512, n_layer=2, grad_clip=5,
                 n_filters=[25, 50, 64], kernel_sizes=[2, 3, 5],
                 stopwords=None, useless_words=None, sess=tf.Session()):
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
        self.stopwords = stopwords
        self.grad_clip = grad_clip

        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes

        self._cursor = None
        self.preprocessing()
        self.build_graph()
    # end constructor


    def build_graph(self):
        with tf.variable_scope('main_model'):
            self.add_input_layer()       
            self.add_word_embedding()
            self.add_concat_conv()
            self.add_lstm_cells()
            self.add_dynamic_rnn()
            self.add_output_layer()
            self.add_backward_path()
        with tf.variable_scope('main_model', reuse=True):
            self.add_inference()
    # end method build_graph


    def add_input_layer(self):
        self.X = tf.placeholder(tf.int32, [None, self.seq_len, self.max_word_len])
        self.Y = tf.placeholder(tf.int32, [None, self.seq_len])
        self.batch_size = tf.placeholder(tf.int32, [])
        self.lr = tf.placeholder(tf.float32) 
        self._cursor = self.X
    # end method add_input_layer


    def add_word_embedding(self):
        # (batch_size, seq_len, max_word_len) -> (batch_size, seq_len, max_word_len, embedding_dims)
        embedding = tf.get_variable('encoder', [self.vocab_char, self.embedding_dims], tf.float32,
                                     tf.random_uniform_initializer(-1.0, 1.0))
        self._cursor = tf.nn.embedding_lookup(embedding, self._cursor)
    # end method add_word_embedding


    def add_concat_conv(self):
        reshaped = tf.reshape(self._cursor, [self.batch_size*self.seq_len, self.max_word_len, self.embedding_dims])
        parallels = []
        for i, (n_filter, kernel_size) in enumerate(zip(self.n_filters, self.kernel_sizes)):
            conv_out = tf.layers.conv1d(inputs = reshaped,
                                        filters = n_filter,
                                        kernel_size  = kernel_size,
                                        padding = 'valid',
                                        use_bias = True,
                                        activation = tf.nn.tanh,
                                        name = 'conv1d'+str(i))
            reduced_len = self.max_word_len - kernel_size + 1
            pool_out = tf.layers.max_pooling1d(inputs = conv_out,
                                               pool_size = reduced_len,
                                               strides = 1,
                                               padding = 'valid')
            parallels.append(tf.reshape(pool_out, [self.batch_size, self.seq_len, n_filter]))
        self._cursor = tf.concat(parallels, 2)
    # end method add_concat_conv


    def add_global_pooling(self):
        Y = tf.layers.max_pooling1d(inputs = self._cursor,
                                    pool_size = self.reduced_len,
                                    strides = 1,
                                    padding = 'valid')
        self._cursor = tf.reshape(Y, [self.batch_size, self.seq_len, self.n_filters])
    # end method add_pooling


    def add_lstm_cells(self):
        def cell():
            cell = tf.nn.rnn_cell.LSTMCell(self.cell_size, initializer=tf.orthogonal_initializer)
            return cell
        self.cells = tf.nn.rnn_cell.MultiRNNCell([cell() for _ in range(self.n_layer)])
    # end method add_rnn_cells


    def add_dynamic_rnn(self):
        self.init_state = self.cells.zero_state(self.batch_size, tf.float32)
        self._cursor, self.final_state = tf.nn.dynamic_rnn(self.cells, self._cursor, initial_state=self.init_state)   
    # end method add_dynamic_rnn


    def add_output_layer(self):
        reshaped = tf.reshape(self._cursor, [-1, self.cell_size])
        self.logits = tf.layers.dense(reshaped, self.vocab_word, name='output')
    # end method add_output_layer


    def add_backward_path(self):
        losses = tf.contrib.seq2seq.sequence_loss(
            logits = tf.reshape(self.logits, [self.batch_size, self.seq_len, self.vocab_word]),
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


    def add_inference(self):
        self.x = tf.placeholder(tf.int32, [1, 1, self.max_word_len])
        self.i_s = self.cells.zero_state(1, tf.float32)
        x_embedded = tf.nn.embedding_lookup(tf.get_variable('encoder'), self.x)
        reshaped = tf.reshape(x_embedded, [1*1, self.max_word_len, self.embedding_dims])

        parallels = []
        for i, (n_filter, kernel_size) in enumerate(zip(self.n_filters, self.kernel_sizes)):
            conv_out = tf.layers.conv1d(inputs = reshaped,
                                        filters = n_filter,
                                        kernel_size  = kernel_size,
                                        padding = 'valid',
                                        use_bias = True,
                                        activation = tf.nn.tanh,
                                        name = 'conv1d'+str(i),
                                        reuse = True)
            reduced_len = self.max_word_len - kernel_size + 1
            pool_out = tf.layers.max_pooling1d(inputs = conv_out,
                                               pool_size = reduced_len,
                                               strides = 1,
                                               padding = 'valid')
            parallels.append(tf.reshape(pool_out, [1, 1, n_filter]))
        rnn_in = tf.concat(parallels, 2)

        rnn_out, self.f_s = tf.nn.dynamic_rnn(self.cells, rnn_in, initial_state=self.i_s)
        logits = tf.layers.dense(tf.reshape(rnn_out, [-1, self.cell_size]), self.vocab_word, name='output',
                                 reuse=True)
        self.y = tf.nn.softmax(logits)
    # end add_sample_model


    def adjust_lr(self, current_step, total_steps):
        max_lr = 0.005
        min_lr = 0.001
        decay_rate = math.log(min_lr/max_lr) / (-total_steps)
        lr = max_lr * math.exp(-decay_rate * current_step)
        return lr
    # end method adjust_lr


    def preprocessing(self):
        text = self.text
        text = text.replace('\n', ' ')

        if self.stopwords is not None:
            if sys.version[0] >= 3:
                table = str.maketrans({stopword: ' '+stopword+' ' for stopword in self.stopwords})
                text = text.translate(table)
            else:
                for stopword in self.stopwords:
                    text = text.replace(stopword, ' '+stopword+' ')

        if self.useless_words is not None:
            if sys.version[0] >= 3:
                table = str.maketrans({useless: ' ' for useless in self.useless_words})
                text = text.translate(table)
            else:
                for useless_word in self.useless_words:
                    text = text.replace(useless_word, '')
        
        text = re.sub('\s+', ' ', text).strip().lower()
        print("Text Cleaned")
        
        chars = set(text)
        self.char2idx = {c: i+1 for i, c in enumerate(chars)}
        self.char2idx['<PAD>'] = 0
        self.idx2char = {i : c for c, i in self.char2idx.items()}
        self.vocab_char = len(self.idx2char)
        print("Vocabulary of Char:", self.vocab_char)

        words = set(text.split())
        self.word2idx = {w: i for i, w in enumerate(words)}
        self.idx2word = {i: w for i, w in enumerate(words)}
        self.max_word_len = max([len(w) for w in words])
        self.vocab_word = len(self.word2idx)
        print("Vocabulary of Word:", self.vocab_word)

        indexed = []
        for word in text.split():
            temp = []
            for char in list(word):
                temp.append(self.char2idx[char])
            if len(temp) < self.max_word_len:
                temp += [0] * (self.max_word_len - len(temp))
            indexed.append(temp)
        self.char_indexed = np.array(indexed) # (None, self.max_word_len)
        print("Char indexed: ", self.char_indexed.shape)

        self.word_indexed = np.array([self.word2idx[word] for word in text.split()])
        print("Word indexed: ", self.word_indexed.shape)
    # end method text_preprocessing


    def next_batch(self, batch_size, text_iter_step):
        window = self.seq_len * batch_size
        for i in range(0, len(self.word_indexed)-window-1, text_iter_step):
            yield (self.char_indexed[i : i+window].reshape(-1, self.seq_len, self.max_word_len),
                   self.word_indexed[i+1 : i+window+1].reshape(-1, self.seq_len))
    # end method next_batch


    def fit(self, prime_texts, text_iter_step=10, n_gen=100, n_epoch=20, batch_size=128,
            en_exp_decay=False):
        global_step = 0
        n_batch = (len(self.word_indexed) - self.seq_len*batch_size - 1) // text_iter_step
        total_steps = n_epoch * n_batch
        self.sess.run(tf.global_variables_initializer()) # initialize all variables
        
        for epoch in range(n_epoch):
            next_state = self.sess.run(self.init_state, {self.batch_size: batch_size})
            for local_step, (X_batch, Y_batch) in enumerate(self.next_batch(batch_size, text_iter_step)):
                lr = self.adjust_lr(global_step, total_steps) if en_exp_decay else 0.001
                _, train_loss, next_state = self.sess.run([self.train_op, self.loss, self.final_state],
                                                          {self.X: X_batch,
                                                           self.Y: Y_batch,
                                                           self.init_state: next_state,
                                                           self.lr: lr,
                                                           self.batch_size: len(X_batch)})
                
                print ('Epoch %d/%d | Batch %d/%d | train loss: %.4f | lr: %.4f'
                        % (epoch+1, n_epoch, local_step, n_batch, train_loss, lr))
                if local_step % 10 == 0:
                    for prime_text in prime_texts:
                        print(self.infer(prime_text, n_gen)+'\n')
                global_step += 1
            
        return log
    # end method fit


    def infer(self, prime_text, n_gen):
        next_state = self.sess.run(self.i_s)
        char_list = [self.char2idx[c] for c in list(prime_text)] + [0] * (self.max_word_len - len(list(prime_text)))
        out_sentence = 'IN: ' + prime_text + '\nOUT: ' + prime_text
        for _ in range(n_gen):
            x = np.reshape(char_list, [1, 1, self.max_word_len])
            softmax_out, next_state = self.sess.run([self.y, self.f_s],
                                                    {self.x: x, self.i_s: next_state})
            probas = softmax_out[0].astype('float64')
            probas = probas / np.sum(probas)
            actions = np.random.multinomial(1, probas, 1)
            idx = np.argmax(actions)
            word = self.idx2word[idx]
            out_sentence = (out_sentence + word) if (word == ',' or word == '.') else (out_sentence + ' ' + word) 
            char_list = [self.char2idx[c] for c in list(word)] + [0] * (self.max_word_len - len(list(word)))
        return out_sentence
    # end method infer
# end class
