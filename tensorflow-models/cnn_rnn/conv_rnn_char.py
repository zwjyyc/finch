import tensorflow as tf
import math
import numpy as np
import sklearn
import string
import re
import collections


class ConvLSTMChar:
    def __init__(self, text, seq_len=50, min_freq=None, stopwords=None, embedding_dims=128,
                 cell_size=64, n_layer=3, clip_grad=5.0, stateful=False,
                 n_filters=64, pool_size=2, kernel_size=5, padding='VALID',
                 sess=tf.Session()):
        """
        Parameters:
        -----------
        sess: object
            tf.Session() object
        text: string
            corpus in one long string, usually obtained by file.read()
        seq_len: int
            Sequence length
        min_freq: int or None
            The minimum char occurence in text required to be saved in vocabulary
            For example, if you pass 10, any char whose occurence below 10 will be indexed as 0
            If you don't want to filter out any word, pass None
        cell_size: int
            Number of units in the rnn cell, default to 128
        n_layers: int
            Number of layers of stacked rnn cells
        stateful: boolean
            Whether the final state of each training batch will be provided as the initial state of next batch
        stopwords: list of characters
            all the stopwords which will be removed from text, usually punctuations
        """
        self.sess = sess
        self.text = text
        self.seq_len = seq_len
        self.min_freq = min_freq
        self.stopwords = stopwords
        self.embedding_dims = embedding_dims

        self.cell_size = cell_size
        self.n_layer = n_layer
        self.stateful = stateful
        self.clip_grad = clip_grad

        self.n_filters = n_filters
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.padding = padding

        self.current_layer = None
        self.current_seq_len = self.seq_len
        
        self.text_preprocessing()
        self.build_graph()
    # end constructor


    def text_preprocessing(self):
        self.clean_text(self.text)
        chars = list(self.text)
        self.build_vocab(chars)
        self.indices = []
        for char in chars:
            try:
                self.indices.append(self.char2idx[char])
            except:
                self.indices.append(0)
    # end method text_preprocessing


    def build_graph(self):
        with tf.variable_scope('main_model'):
            self.add_input_layer()       
            self.add_word_embedding()

            self.add_conv1d('conv', filter_shape=[self.kernel_size, self.embedding_dims, self.n_filters])
            self.add_maxpool(self.pool_size)

            self.add_lstm_cells()
            self.add_dynamic_rnn()
            
            self.add_output_layer()
            self.add_backward_path()
    # end method build_graph


    def add_input_layer(self):
        self.X = tf.placeholder(tf.int32, [None, self.seq_len])
        self.Y = tf.placeholder(tf.int32, [None, self.vocab_size]) 
        self.batch_size = tf.placeholder(tf.int32)
        self.lr = tf.placeholder(tf.float32)
        self.current_layer = self.X
    # end method add_input_layer


    def add_word_embedding(self):
        # (batch_size, seq_len) -> (batch_size, seq_len, n_hidden)
        E = tf.get_variable('E', [self.vocab_size, self.embedding_dims], tf.float32, tf.random_normal_initializer())
        E = tf.nn.embedding_lookup(E, self.current_layer)
        self.current_layer = E
    # end method add_word_embedding


    def add_conv1d(self, name, filter_shape, stride=1):
        W = tf.get_variable(name+'_W', filter_shape, tf.float32, tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable(name+'_b', filter_shape[-1], tf.float32, tf.constant_initializer(0.1))
        conv = tf.nn.conv1d(self.current_layer, W, stride=stride, padding=self.padding)
        conv = tf.nn.bias_add(conv, b)
        conv = tf.nn.relu(conv)
        self.current_layer = conv
        if self.padding == 'VALID':
            self.current_seq_len = int((self.current_seq_len - self.kernel_size + 1) / stride)
        if self.padding == 'SAME':
            self.current_seq_len = int(self.current_seq_len / stride)
    # end method add_conv1d



    def add_maxpool(self, k=2):
        conv = tf.expand_dims(self.current_layer, 1)
        conv = tf.nn.max_pool(conv, ksize=[1,1,k,1], strides=[1,1,k,1], padding=self.padding)
        conv = tf.squeeze(conv)
        self.current_seq_len = int(self.current_seq_len / k)
        # reshape to produce explicit tensor shape, because rnn will reject unknown shape
        self.current_layer = tf.reshape(conv, [self.batch_size, self.current_seq_len, self.n_filters])
    # end method add_maxpool


    def add_lstm_cells(self):
        def cell():
            cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size)
            return cell
        self.cells = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(self.n_layer)])
    # end method add_rnn_cells


    def add_dynamic_rnn(self):
        self.init_state = self.cells.zero_state(self.batch_size, tf.float32)
        self.current_layer, self.final_state = tf.nn.dynamic_rnn(self.cells, self.current_layer,
                                                                 initial_state=self.init_state,
                                                                 time_major=False)    
    # end method add_dynamic_rnn


    def add_output_layer(self):
        time_major = tf.unstack(tf.transpose(self.current_layer, [1,0,2]))
        W = tf.get_variable('logits_W', [self.cell_size, self.vocab_size], tf.float32,
                             tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable('logits_b', [self.vocab_size], tf.float32, tf.constant_initializer(0.1))
        self.logits = tf.nn.bias_add(tf.matmul(time_major[-1], W), b)
        self.softmax_out = tf.nn.softmax(self.logits)
    # end method add_output_layer


    def add_backward_path(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        # gradient clipping
        gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tf.trainable_variables()), self.clip_grad)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))
    # end method add_backward_path


    def decrease_lr(self, en_exp_decay, global_step, n_epoch, nb_batch):
        if en_exp_decay:
            max_lr = 0.003
            min_lr = 0.0005
            decay_rate = math.log(min_lr/max_lr) / (-n_epoch*nb_batch)
            lr = max_lr*math.exp(-decay_rate*global_step)
        else:
            lr = 0.001
        return lr
    # end method adjust_lr


    def clean_text(self, text):
        text = text.replace('\n', ' ')
        if self.stopwords is None:
            self.stopwords = string.punctuation
        text = re.sub(r'[{}]'.format(''.join(self.stopwords)), ' ', text)
        text = re.sub('\s+', ' ', text ).strip().lower()
        self.text = text
    # end method clean_text


    def build_vocab(self, chars):
        char_freqs = collections.Counter(chars)
        n_total = len(char_freqs)
        if self.min_freq is not None:
            char_freqs = {char:freq for char,freq in char_freqs.items() if freq > self.min_freq}
        chars = char_freqs.keys()
        self.char2idx = {char:(idx+1) for idx,char in enumerate(chars)} # create word -> index mapping
        self.char2idx['_unknown'] = 0 # add unknown key -> 0 index
        self.idx2char = {idx:char for char,idx in self.char2idx.items()} # create index -> word mapping
        assert len(self.idx2char) == len(self.char2idx), "len(idx2char) != len(char2idx)"
        self.vocab_size = len(self.idx2char)
        print('Vocabulary size:', self.vocab_size, '/', n_total)
    # end method build_vocab


    def fit_text(self, text_iter_step=1, temperature=1.0, n_gen=100, n_epoch=50, batch_size=128,
                 en_exp_decay=True, en_shuffle=True):
        
        window = self.seq_len + 1
        X = np.array([self.indices[i:i+window] for i in range(0, len(self.indices)-window, text_iter_step)])
        Y = X[:, -1]
        X = X[:, :-1]
        Y = tf.contrib.keras.utils.to_categorical(Y)
        print('X shape:', X.shape, '|', 'Y shape:', Y.shape)
        
        log = {'loss': []}
        global_step = 0
        self.sess.run(tf.global_variables_initializer()) # initialize all variables
        
        for epoch in range(n_epoch):
            next_state = self.sess.run(self.init_state, feed_dict={self.batch_size:batch_size})
            batch_count = 1
            if en_shuffle:
                X, Y = sklearn.utils.shuffle(X, Y)
            for X_batch, Y_batch in zip(self.gen_batch(X, batch_size), self.gen_batch(Y, batch_size)):
                lr = self.decrease_lr(en_exp_decay, global_step, n_epoch, int(len(X)/batch_size))
                if (self.stateful) and (len(X_batch) == batch_size):
                    _, loss, next_state = self.sess.run([self.train_op, self.loss, self.final_state],
                                                        {self.X: X_batch, self.Y: Y_batch,
                                                         self.init_state: next_state,
                                                         self.batch_size: len(X_batch),
                                                         self.lr: lr})
                else:
                    _, loss = self.sess.run([self.train_op, self.loss],
                                            {self.X: X_batch, self.Y: Y_batch,
                                             self.batch_size: len(X_batch),
                                             self.lr: lr})
                if batch_count % 10 == 0:
                    print ('Epoch %d/%d | Batch %d/%d | train loss: %.4f | lr: %.4f'
                            % (epoch+1, n_epoch, batch_count, (len(X)/batch_size), loss, lr))
                if batch_count % 100 == 0:
                    print(self.sample(temperature, n_gen), end='\n\n')
                log['loss'].append(loss)
                batch_count += 1
                global_step += 1
            
        return log
    # end method fit


    def sample(self, temperature, n_gen):
        while True:
            random_start = np.random.randint(0, len(self.text)-1-self.seq_len)
            prime_text = self.text[random_start: random_start + self.seq_len]
            try:
                x = [self.char2idx[char] for char in list(prime_text)]
            except:
                continue
            else:
                break
        
        out_sentence = 'IN: ' + prime_text + '\nOUT: ' + prime_text
        next_state = self.sess.run(self.init_state, feed_dict={self.batch_size: 1})
        for _ in range(n_gen):
            softmax_out, next_state = self.sess.run([self.softmax_out, self.final_state],
                                                    {self.X: np.atleast_2d(x),
                                                     self.init_state: next_state,
                                                     self.batch_size: 1})
            idx = self.infer_idx(softmax_out[0], temperature)
            if idx == 0:
                break
            char = self.idx2char[idx]
            out_sentence = out_sentence + char
            x = x[1:]
            x.append(idx)
        return out_sentence
    # end method sample


    def infer_idx(self, preds, temperature): # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
    # end method infer_idx


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method gen_batch
# end class
