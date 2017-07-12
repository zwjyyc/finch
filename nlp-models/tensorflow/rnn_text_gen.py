import tensorflow as tf
import math
import numpy as np
import re


class RNNTextGen:
    def __init__(self, text, seq_len=50, embedding_dims=128, cell_size=512, n_layer=2, sess=tf.Session()):
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
            self.add_inference()
    # end method build_graph


    def add_input_layer(self):
        self.X = tf.placeholder(tf.int32, [None, self.seq_len])
        self.Y = tf.placeholder(tf.int32, [None, self.seq_len])
        self.batch_size = tf.placeholder(tf.int32, [])
        self.lr = tf.placeholder(tf.float32) 
        self._cursor = self.X
    # end method add_input_layer


    def add_word_embedding(self):
        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_dims)
        embedding = tf.get_variable('encoder', [self.vocab_size, self.embedding_dims], tf.float32,
                                     tf.random_uniform_initializer(-1.0, 1.0))
        self._cursor = tf.nn.embedding_lookup(embedding, self._cursor)
    # end method add_word_embedding


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
        optimizer = tf.train.AdamOptimizer(self.lr)
        gradients = optimizer.compute_gradients(self.loss)
        clipped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        self.train_op = optimizer.apply_gradients(clipped_gradients)
    # end method add_backward_path


    def add_inference(self):
        self.x = tf.placeholder(tf.int32, [1, 1])
        self.i_s = self.cells.zero_state(1, tf.float32)
        x_embedded = tf.nn.embedding_lookup(tf.get_variable('encoder'), self.x)
        y, self.f_s = tf.nn.dynamic_rnn(self.cells, x_embedded, initial_state=self.i_s)
        y = tf.layers.dense(tf.reshape(y, [-1, self.cell_size]), self.vocab_size, name='output', reuse=True)
        self.y = tf.nn.softmax(y)
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
        text = re.sub('\s+', ' ', text).strip().lower()
        
        chars = set(text)
        self.char2idx = {c: i for i, c in enumerate(chars)}
        self.idx2char = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(self.idx2char)
        print('Vocabulary size:', self.vocab_size)

        self.indexed = np.array([self.char2idx[char] for char in list(text)])
    # end method text_preprocessing


    def next_batch(self, batch_size, text_iter_step):
        window = self.seq_len * batch_size
        for i in range(0, len(self.indexed)-window-1, text_iter_step):
            yield (self.indexed[i : i+window].reshape(-1, self.seq_len),
                   self.indexed[i+1 : i+window+1].reshape(-1, self.seq_len))
    # end method next_batch


    def fit(self, prime_texts, text_iter_step=10, n_gen=500, n_epoch=20, batch_size=128,
            en_exp_decay=False):
        global_step = 0
        n_batch = (len(self.indexed) - self.seq_len*batch_size - 1) // text_iter_step
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
                if local_step % 10 == 0:
                    print ('Epoch %d/%d | Batch %d/%d | train loss: %.4f | lr: %.4f'
                            % (epoch+1, n_epoch, local_step, n_batch, train_loss, lr))
                if local_step % 100 == 0:
                    for prime_text in prime_texts:
                        print(self.infer(prime_text, n_gen)+'\n')
                global_step += 1
            
        return log
    # end method fit


    def infer(self, prime_text, n_gen):
        # warming up
        next_state = self.sess.run(self.i_s)
        char_list = list(prime_text)
        for char in char_list[:-1]:
            x = np.atleast_2d(self.char2idx[char]) 
            next_state = self.sess.run(self.f_s, {self.x:x, self.i_s:next_state})
        # end warming up

        out_sentence = 'IN: ' + prime_text + '\nOUT: ' + prime_text
        char = char_list[-1]
        for _ in range(n_gen):
            x = np.atleast_2d(self.char2idx[char])
            softmax_out, next_state = self.sess.run([self.y, self.f_s],
                                                    {self.x:x, self.i_s:next_state})
            probas = softmax_out[0].astype('float64')
            probas = probas / np.sum(probas)
            actions = np.random.multinomial(1, probas, 1)
            idx = np.argmax(actions)
            char = self.idx2char[idx]
            out_sentence = out_sentence + char
        return out_sentence
    # end method infer
# end class
