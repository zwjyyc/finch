import tensorflow as tf
import math
import numpy as np
import sklearn


class RNNTextGen:
    def __init__(self, seq_len, vocab_size, cell_size, n_layers, resolution, word2idx, idx2word, sess):
        """
        Parameters:
        -----------
        seq_len: int
            Sequence length
        vocab_size: int
            Vocabulary size
        cell_size: int
            Number of units in the rnn cell
        n_layers: int
            Number of layers of stacked rnn cells
        resolution: string
            word-level or char-level
        word2idx: dict
            {word: index}
        idx2word: dict
            {index: word}
        sess: object
            tf.Session() object 
        """
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.cell_size = cell_size
        self.n_layers = n_layers
        self.resolution = resolution
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.sess = sess
        self.current_layer = None
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.add_input_layer()
                    
        self.add_word_embedding_layer()
        self.add_lstm_cells()
        self.add_dynamic_rnn()
        self.reshape_rnn_out()

        self.add_output_layer()
        self.add_backward_path()
    # end method build_graph


    def add_input_layer(self):
        self.batch_size = tf.placeholder(tf.int32)
        self.X = tf.placeholder(tf.int32, [None, self.seq_len])
        self.Y = tf.placeholder(tf.int32, [None, self.seq_len])
        self.W = tf.get_variable('W', [self.cell_size, self.vocab_size], tf.float32,
                                 tf.contrib.layers.variance_scaling_initializer())
        self.b = tf.get_variable('b', [self.vocab_size], tf.float32, tf.constant_initializer(0.0))
        self.current_layer = self.X
    # end method add_input_layer


    def add_word_embedding_layer(self):
        """
        X from (batch_size, seq_len) -> (batch_size, seq_len, n_hidden)
        where each word in (batch_size, seq_len) is represented by a vector of length [n_hidden]
        """
        embedding_mat = tf.get_variable('embedding_mat', [self.vocab_size, self.cell_size], tf.float32,
                                        tf.random_normal_initializer())
        embedding_out = tf.nn.embedding_lookup(embedding_mat, self.current_layer)
        self.current_layer = embedding_out
    # end method add_word_embedding_layer


    def add_lstm_cells(self):
        cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size)
        self.cells = tf.contrib.rnn.MultiRNNCell([cell] * self.n_layers)
    # end method add_rnn_cells


    def add_dynamic_rnn(self):
        self.init_state = self.cells.zero_state(self.batch_size, tf.float32)
        self.current_layer, self.final_state = tf.nn.dynamic_rnn(self.cells, self.current_layer,
                                                                 initial_state=self.init_state,
                                                                 time_major=False)    
    # end method add_dynamic_rnn


    def reshape_rnn_out(self):
        self.current_layer = tf.reshape(self.current_layer, [-1, self.cell_size])
    # end method add_rnn_out


    def add_output_layer(self):
        self.logits = tf.nn.bias_add(tf.matmul(self.current_layer, self.W), self.b)
        self.softmax_out = tf.nn.softmax(self.logits)
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
        self.lr = tf.placeholder(tf.float32)
        gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tf.trainable_variables()), 4.5)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))
    # end method add_backward_path


    def decrease_lr(self, en_exp_decay, global_step, n_epoch, nb_batch):
        if en_exp_decay:
            max_lr = 0.003
            min_lr = 0.0001
            decay_rate = math.log(min_lr/max_lr) / (-n_epoch*nb_batch)
            lr = max_lr*math.exp(-decay_rate*global_step)
        else:
            lr = 0.0005
        return lr
    # end method adjust_lr


    def fit(self, X, n_epoch=10, batch_size=128, en_exp_decay=True, en_shuffle=False,
            sample_model=None, prime_texts=None, num_gen=None, temperature=1.0):
        print('Training', len(X), 'samples')
        log = {'loss': []}
        global_step = 0
        self.sess.run(tf.global_variables_initializer()) # initialize all variables
        
        for epoch in range(n_epoch):
            next_state = self.sess.run(self.init_state, feed_dict={self.batch_size:batch_size})
            batch_count = 1
            if en_shuffle:
                X = sklearn.utils.shuffle(X)
            for X_batch in self.gen_batch(X, batch_size):
                Y_batch = np.roll(X_batch, -1, axis=1)
                lr = self.decrease_lr(en_exp_decay, global_step, n_epoch, int(len(X)/batch_size))
                if len(X_batch) == batch_size:
                    _, loss, next_state = self.sess.run([self.train_op, self.loss, self.final_state],
                                                         feed_dict={self.X:X_batch, self.Y:Y_batch,
                                                                    self.init_state:next_state,
                                                                    self.batch_size:len(X_batch), self.lr:lr})
                else:
                    _, loss = self.sess.run([self.train_op, self.loss],
                                             feed_dict={self.X:X_batch, self.Y:Y_batch,
                                                        self.batch_size:len(X_batch), self.lr:lr})
                if batch_count % 10 == 0:
                    print ('Epoch %d/%d | Batch %d/%d | train loss: %.4f | lr: %.4f' % (epoch+1, n_epoch,
                    batch_count, (len(X)/batch_size), loss, lr))
                log['loss'].append(loss)
                batch_count += 1
                global_step += 1
            
            if sample_model is not None:
                for prime_text in prime_texts:
                    print(self.sample(sample_model, prime_text, num_gen, temperature), end='\n\n')
            
        return log
    # end method fit


    def sample(self, sample_model, prime_text, num_gen, temperature):
        # warming up
        next_state = self.sess.run(sample_model.init_state, feed_dict={sample_model.batch_size:1})
        if self.resolution == 'word':
            word_list = prime_text.split()
        if self.resolution == 'char':
            word_list = list(prime_text)
        for word in word_list[:-1]:
            x = np.zeros([1,1])
            x[0,0] = self.word2idx[word] 
            next_state = self.sess.run(sample_model.final_state, feed_dict={sample_model.X:x,
                                                                            sample_model.init_state:next_state})
        # end warming up

        out_sentence = prime_text + '|'
        word = word_list[-1]
        for n in range(num_gen):
            x = np.zeros([1,1])
            x[0,0] = self.word2idx[word]
            softmax_out, next_state = self.sess.run([sample_model.softmax_out, sample_model.final_state],
                                                     feed_dict={sample_model.X:x,
                                                                sample_model.init_state:next_state})
            idx = self.infer_idx(softmax_out[0], temperature)
            if idx == 0:
                break
            word = self.idx2word[idx]
            if self.resolution == 'word':
                out_sentence = out_sentence + ' ' + word
            if self.resolution == 'char':
                out_sentence = out_sentence + word
        return(out_sentence)
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
