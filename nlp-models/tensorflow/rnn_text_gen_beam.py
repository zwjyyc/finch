import tensorflow as tf
import numpy as np
import math


class RNNTextGen:
    def __init__(self, text, seq_len=50, embedding_dims=128, cell_size=256, n_layer=2, grad_clip=5., 
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
        self.grad_clip = grad_clip
        self._pointer = None
        self.preprocessing()
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.add_input_layer()       
        self.add_word_embedding()
        self.add_lstm_cells()
        self.add_dynamic_rnn()
        self.add_output_layer()
        self.add_backward_path()
    # end method build_graph


    def add_input_layer(self):
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, None])
        self.batch_size = tf.placeholder(tf.int32, [])
        self.lr = tf.placeholder(tf.float32) 
        self._pointer = self.X
    # end method add_input_layer


    def add_word_embedding(self):
        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_dims)
        embedding = tf.get_variable('encoder', [self.vocab_size, self.embedding_dims], tf.float32,
                                     tf.random_uniform_initializer(-1.0, 1.0))
        self._pointer = tf.nn.embedding_lookup(embedding, self._pointer)
    # end method add_word_embedding


    def add_lstm_cells(self):
        lstm = lambda x : tf.nn.rnn_cell.LSTMCell(x, initializer=tf.orthogonal_initializer())
        self.cells = tf.nn.rnn_cell.MultiRNNCell([lstm(self.cell_size) for _ in range(self.n_layer)])
    # end method add_rnn_cells


    def add_dynamic_rnn(self):
        self.init_state = self.cells.zero_state(self.batch_size, tf.float32)
        self._pointer, self.final_state = tf.nn.dynamic_rnn(self.cells, self._pointer, initial_state=self.init_state)   
    # end method add_dynamic_rnn


    def add_output_layer(self):
        reshaped = tf.reshape(self._pointer, [-1, self.cell_size])
        self.logits = tf.layers.dense(reshaped, self.vocab_size)
        self.softmax_out = tf.nn.softmax(self.logits)
    # end method add_output_layer


    def add_backward_path(self):
        losses = tf.contrib.seq2seq.sequence_loss(
            logits = tf.reshape(self.logits, [self.batch_size, self.seq_len, self.vocab_size]),
            targets = self.Y,
            weights = tf.ones([self.batch_size, self.seq_len]),
            average_across_timesteps = True,
            average_across_batch = True)
        self.loss = tf.reduce_sum(losses)
        # gradient clipping
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
        self.train_op = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(clipped_gradients, params))
    # end method add_backward_path


    def adjust_lr(self, current_step, total_steps):
        max_lr = 0.003
        min_lr = 0.0001
        decay_rate = math.log(min_lr/max_lr) / (-total_steps)
        lr = max_lr * math.exp(-decay_rate * current_step)
        return lr
    # end method adjust_lr


    def preprocessing(self):
        text = self.text 
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


    def fit(self, start_word, text_iter_step=1, n_gen=500, n_epoch=1, batch_size=128, en_exp_decay=False):
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
                    print(self.infer(start_word, n_gen)+'\n')
                global_step += 1
    # end method fit


    def infer(self, start_word, n_gen, beam_width=2):
        # warming up
        next_state = self.sess.run(self.init_state, {self.batch_size: 1})
        char_list = list(start_word)
        for char in char_list[:-1]:
            next_state = self.sess.run(self.final_state, {self.X: np.atleast_2d(self.char2idx[char]),
                                                          self.init_state: next_state})
        
        next_states = [next_state] * beam_width
        joint_probas = [np.log(1.)] * beam_width
        out_sentences = ['\n[IN]: '+start_word+'\n[OUT]: '+start_word] * beam_width
        chars = [char] * beam_width

        for i in range(beam_width):
            softmax_out, next_states[i] = self.sess.run([self.softmax_out, self.final_state],
                                                        {self.X: np.atleast_2d(self.char2idx[chars[i]]),
                                                         self.init_state: next_states[i]})
            softmax_out = softmax_out.ravel()
            i_largest_idx = list(reversed(softmax_out.argsort()))[i]
            joint_probas[i] += np.log(softmax_out[i_largest_idx])
            char = self.idx2char[i_largest_idx]
            out_sentences[i] = out_sentences[i] + char
            chars[i] = char

            for _ in range(n_gen):
                softmax_out, next_states[i] = self.sess.run([self.softmax_out, self.final_state],
                                                            {self.X: np.atleast_2d(self.char2idx[chars[i]]),
                                                             self.init_state: next_states[i]})
                probas = softmax_out.ravel().astype(np.float64)
                probas = probas / np.sum(probas)
                actions = np.random.multinomial(1, probas, 1)
                idx = np.argmax(actions)
                joint_probas[i] += np.log(probas[idx])
                char = self.idx2char[idx]
                chars[i] = char
                out_sentences[i] = out_sentences[i] + char
        
        higher_one = np.argmax(joint_probas)
        proba_verbose = ''
        for i in range(beam_width):
            proba_verbose = proba_verbose + 'Beam%d: %d | '%(i, joint_probas[i])
        return out_sentences[higher_one] + '\n[Beam%d Selected] | '%(higher_one) + proba_verbose
    # end method infer
# end class
