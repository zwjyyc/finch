from tensorflow.python.layers import core as core_layers
import tensorflow as tf
import numpy as np


class RNN_VAE:
    def __init__(self, rnn_size, n_layers, X_word2idx, embedding_dim, sess=tf.Session(), grad_clip=5.0):
        self.rnn_size = rnn_size
        self.n_layers = n_layers
        self.grad_clip = grad_clip
        self.X_word2idx = X_word2idx
        self.embedding_dim = embedding_dim
        self.sess = sess
        self.register_symbols()
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.add_input_layer()
        self.add_encoder_layer()
        self.add_latent_layer()
        self.add_decoder_layer()
        self.add_backward_path()
    # end method


    def add_input_layer(self):
        self.X_in = tf.placeholder(tf.int32, [None, None])
        self.X_out = tf.placeholder(tf.int32, [None, None])
        self.X_seq_len = tf.placeholder(tf.int32, [None])
        self.batch_size = tf.shape(self.X_in)[0]
    # end method


    def lstm_cell(self, reuse=False):
        return tf.nn.rnn_cell.LSTMCell(self.rnn_size, initializer=tf.orthogonal_initializer(), reuse=reuse)
    # end method


    def _attention(self, reuse=False):
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units = self.rnn_size, 
            memory = self.encoder_out,
            memory_sequence_length = self.X_seq_len)
        
        return tf.contrib.seq2seq.AttentionWrapper(
            cell = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell(reuse) for _ in range(self.n_layers)]),
            attention_mechanism = attention_mechanism,
            attention_layer_size = self.rnn_size)
    # end method


    def add_encoder_layer(self):
        self.encoder_embedding = tf.get_variable('encoder_embedding', [len(self.X_word2idx), self.embedding_dim],
                                                  tf.float32, tf.random_normal_initializer())            
        self.encoder_out, self.encoder_state = tf.nn.dynamic_rnn(
            cell = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell() for _ in range(self.n_layers)]), 
            inputs = tf.nn.embedding_lookup(self.encoder_embedding, self.X_in),
            sequence_length = self.X_seq_len,
            dtype = tf.float32)
    # end method
    

    def add_latent_layer(self):
        self.mean = tf.layers.dense(self.encoder_state[-1].h, self.rnn_size)
        self.gamma = tf.layers.dense(self.encoder_state[-1].h, self.rnn_size)
        sigma = tf.exp(0.5 * self.gamma)
        self.latent_vector = self.mean + sigma * tf.random_normal(tf.shape(self.gamma))
        self.decoder_init_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(
            self.latent_vector, self.latent_vector) for _ in range(self.n_layers)])
    # end method
    

    def add_decoder_layer(self):
        decoder_embedding = self.encoder_embedding
        self.decoder_seq_len = self.X_seq_len + 1 # because of go and eos symbol in decoder stage
        
        with tf.variable_scope('decode'):
            decoder_cell = self._attention()
            training_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                inputs = tf.nn.embedding_lookup(decoder_embedding, self.processed_decoder_input()),
                sequence_length = self.decoder_seq_len,
                embedding = decoder_embedding,
                sampling_probability = 0.2,
                time_major = False)
            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell = decoder_cell,
                helper = training_helper,
                initial_state = decoder_cell.zero_state(self.batch_size, tf.float32).clone(
                    cell_state=self.decoder_init_state),
                output_layer = core_layers.Dense(len(self.X_word2idx)))
            training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = training_decoder,
                impute_finished = True,
                maximum_iterations = tf.reduce_max(self.decoder_seq_len))
            self.training_logits = training_decoder_output.rnn_output
        
        with tf.variable_scope('decode', reuse=True):
            decoder_cell = self._attention(reuse=True)
            predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding = decoder_embedding,
                start_tokens = tf.tile(tf.constant([self._x_go], dtype=tf.int32), tf.shape(self.X_seq_len)),
                end_token = self._x_eos)
            predicting_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell = decoder_cell,
                helper = predicting_helper,
                initial_state = decoder_cell.zero_state(self.batch_size, tf.float32).clone(
                    cell_state=self.decoder_init_state),
                output_layer = core_layers.Dense(len(self.X_word2idx), _reuse=True))
            predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = predicting_decoder,
                impute_finished = True,
                maximum_iterations = tf.reduce_max(self.decoder_seq_len) * 2)
            self.predicting_ids = predicting_decoder_output.sample_id
    # end method


    def add_backward_path(self):
        masks = tf.sequence_mask(self.decoder_seq_len, tf.reduce_max(self.decoder_seq_len), dtype=tf.float32)
        self.gen_loss = tf.contrib.seq2seq.sequence_loss(logits = self.training_logits,
                                                         targets = self.X_out,
                                                         weights = masks)
        # the kl divergence between the gaussian distribution and the actual distribution of coding
        self.latent_loss = 0.5 * tf.reduce_sum(tf.exp(self.gamma) + tf.square(self.mean) - 1 - self.gamma)
        loss = self.gen_loss + self.latent_loss
        # gradient clipping
        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
        self.train_op = tf.train.AdamOptimizer().apply_gradients(zip(clipped_gradients, params))
    # end method


    def processed_decoder_input(self):
        decoder_input = tf.concat([tf.fill([self.batch_size, 1], self._x_go), self.X_in], 1)
        return decoder_input
    # end method


    def pad_sentence_batch(self, sentence_batch, mode):
        padded_seqs = []
        seq_lens = []
        max_sentence_len = max([len(sentence) for sentence in sentence_batch])
        for sentence in sentence_batch:
            if mode == 'decoder_in':
                padded_seqs.append(sentence + [self._x_pad] * (max_sentence_len - len(sentence)))
            if mode == 'decoder_out':
                padded_seqs.append(sentence + [self._x_eos] + [self._x_pad] * (max_sentence_len - len(sentence)))
            seq_lens.append(len(sentence))
        return padded_seqs, seq_lens
    # end method


    def next_batch(self, X, batch_size):
        for i in range(0, len(X) - len(X) % batch_size, batch_size):
            X_batch = X[i : i + batch_size]
            padded_X_in_batch, X_batch_lens = self.pad_sentence_batch(X_batch, 'decoder_in')
            padded_X_out_batch, _ = self.pad_sentence_batch(X_batch, 'decoder_out')
            yield (np.array(padded_X_in_batch), np.array(padded_X_out_batch), X_batch_lens)
    # end method


    def fit(self, X_train, X_test, n_epoch=60, display_step=50, batch_size=128):
        X_test_in_batch, X_test_out_batch, X_test_batch_lens = next(self.next_batch(X_test, batch_size))
        self.sess.run(tf.global_variables_initializer())
        
        for epoch in range(1, n_epoch+1):
            for i, (X_train_in_batch, X_train_out_batch, X_train_batch_lens) in enumerate(self.next_batch(X_train, batch_size)):
                _, gen_loss, latent_loss = self.sess.run([self.train_op, self.gen_loss, self.latent_loss],
                    {self.X_in: X_train_in_batch, self.X_out: X_train_out_batch, self.X_seq_len: X_train_batch_lens})
                if i % display_step == 0:
                    val_gen_loss = self.sess.run(self.gen_loss,
                        {self.X_in: X_test_in_batch, self.X_out:X_test_out_batch, self.X_seq_len: X_test_batch_lens})
                    print("Epoch %d/%d | Batch %d/%d | train_gen_loss: %.3f, train_latent_loss: %.3f | test_gen_loss: %.3f"
                        % (epoch, n_epoch, i, len(X_train)//batch_size, gen_loss, latent_loss, val_gen_loss))
    # end method


    def infer(self, input_word, X_idx2word):        
        input_indices = [self.X_word2idx.get(char, self._x_unk) for char in input_word]
        out_indices = self.sess.run(self.predicting_ids, {
            self.X_in: np.atleast_2d(input_indices), self.X_seq_len: np.atleast_1d(len(input_indices))})[0]
        print('\nI: {}'.format(' '.join([X_idx2word[i] for i in input_indices])))
        print('O: {}'.format(' '.join([X_idx2word[i] for i in out_indices])))
    # end method


    def register_symbols(self):
        self._x_go = self.X_word2idx['<GO>']
        self._x_eos = self.X_word2idx['<EOS>']
        self._x_pad = self.X_word2idx['<PAD>']
        self._x_unk = self.X_word2idx['<UNK>']
    # end method
# end class