import tensorflow as tf
import numpy as np
import os, sys
from tensorflow.python.layers import core as core_layers


class VRAE:
    def __init__(
        self, text, seq_len=50, embedding_dims=128, rnn_size=256, n_layers=1, grad_clip=5, beam_width=5,
        latent_size=16, anneal_bias=6000, anneal_max=1.0, word_dropout_rate=0.2, mutinfo_w=1.0,
        sess=tf.Session()):
        
        self.sess = sess
        self.text = text
        self.seq_len = seq_len
        self.embedding_dims = embedding_dims
        self.rnn_size = rnn_size
        self.n_layers = n_layers
        self.grad_clip = grad_clip
        self.beam_width = beam_width

        self.latent_size = latent_size
        self.anneal_bias = anneal_bias
        self.anneal_max = anneal_max
        self.word_dropout_rate = word_dropout_rate
        self.mutinfo_w = mutinfo_w
        
        self.preprocessing()
        self.build_graph()
        self.saver = tf.train.Saver()
        self.model_path = './saved/'+sys.argv[0][:-3]
    # end constructor


    def build_graph(self):
        self.add_input_layer()
        self.add_encoder()
        self.add_latent_layer()       
        self.add_decoder()
        self.add_backward_path()
    # end method


    def add_input_layer(self):
        self.sequence = tf.placeholder(tf.int32, [None, None])
        self.sequence_dropped = tf.placeholder(tf.int32, [None, None])
        self.sequence_length = tf.placeholder(tf.int32, [None])
        self._batch_size = tf.shape(self.sequence)[0]
        self.gen_seq_length = tf.placeholder(tf.int32, [])
    # end method


    def add_encoder(self):
        with tf.variable_scope('encoder'):
            (out_fw, out_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = self._rnn_cell(),
                cell_bw = self._rnn_cell(), 
                inputs = tf.contrib.layers.embed_sequence(
                    self.sequence, self.vocab_size, self.embedding_dims),
                sequence_length = self.sequence_length,
                dtype = tf.float32)
            self._encoded_vec = tf.concat((state_fw, state_bw), -1)
    # end method


    def add_latent_layer(self):
        self._mean = tf.layers.dense(self._encoded_vec, self.latent_size, tf.nn.elu)
        self._gamma = tf.layers.dense(self._encoded_vec, self.latent_size, tf.nn.elu)
        _noise = tf.truncated_normal(tf.shape(self._gamma))
        self._latent_vec = self._mean + tf.exp(0.5 * self._gamma) * _noise
        state = tf.layers.dense(self._latent_vec, self.rnn_size, tf.nn.elu)
        self._decoder_init = tuple([state * self.n_layers])
    # end method


    def add_decoder(self):
        with tf.variable_scope('decode'): # decode (training)
            decoder_embedding = tf.get_variable('decoder_embedding', [self.vocab_size, self.embedding_dims],
                                                 tf.float32, tf.random_uniform_initializer(-1.0, 1.0))
            decoder_cell = tf.nn.rnn_cell.MultiRNNCell([self._rnn_cell() for _ in range(self.n_layers)])

            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs = tf.nn.embedding_lookup(decoder_embedding, self._decoder_input()),
                sequence_length = self.sequence_length+1,
                time_major = False)
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell = decoder_cell,
                helper = helper,
                initial_state = self._decoder_init,
                output_layer = core_layers.Dense(self.vocab_size))
            decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = decoder,
                impute_finished = True,
                maximum_iterations = tf.reduce_max(self.sequence_length+1))
            self.logits = decoder_output.rnn_output
        
        with tf.variable_scope('decode', reuse=True): # decode (predicting)
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell = tf.nn.rnn_cell.MultiRNNCell(
                    [self._rnn_cell(reuse=True) for _ in range(self.n_layers)]),
                embedding = tf.get_variable('decoder_embedding'),
                start_tokens = tf.tile(tf.constant(
                    [self.char2idx['<start>']], dtype=tf.int32), [self._batch_size]),
                end_token = self.char2idx['<end>'],
                initial_state = tf.contrib.seq2seq.tile_batch(self._decoder_init, self.beam_width),
                beam_width = self.beam_width,
                output_layer = core_layers.Dense(self.vocab_size, _reuse=True),
                length_penalty_weight = 0.0)
            decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = decoder,
                impute_finished = False,
                maximum_iterations = self.gen_seq_length)
            self.predicted_ids = decoder_output.predicted_ids[:, :, 0]
    # end method


    def add_backward_path(self):
        global_step = tf.Variable(0, trainable=False)

        targets = self._decoder_output()
        self.nll_loss = tf.reduce_sum(tf.contrib.seq2seq.sequence_loss(
            logits = self.logits,
            targets = targets,
            weights = tf.cast(tf.ones_like(targets), tf.float32),
            average_across_timesteps = False,
            average_across_batch = True))

        _batch_size = tf.cast(self._batch_size, tf.float32)
        self.kl_w = self.anneal_max * tf.sigmoid((10 / self.anneal_bias) * (
            tf.cast(global_step, tf.float32) - tf.constant(self.anneal_bias / 2)))
        self.kl_loss = 0.5 * tf.reduce_sum(
            tf.exp(self._gamma) + tf.square(self._mean) - 1 - self._gamma) / _batch_size
        self.mutinfo_loss = tf.reduce_sum(self._mutinfo_loss(
            self._latent_vec, self._mean, self._gamma)) / _batch_size

        loss_op = self.nll_loss + self.kl_w * (self.kl_loss + self.mutinfo_w * self.mutinfo_loss) 
        # gradient clipping
        params = tf.trainable_variables()
        gradients = tf.gradients(loss_op, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
        self.train_op = tf.train.AdamOptimizer().apply_gradients(
            zip(clipped_gradients, params), global_step=global_step)
    # end method


    def preprocessing(self):
        text = self.text 
        chars = set(text)
        self.char2idx = {c: i+3 for i, c in enumerate(chars)}
        self.char2idx['<start>'] = 0
        self.char2idx['<end>'] = 1
        self.char2idx['<unk>'] = 2
        self.vocab_size = len(self.char2idx)
        print('Vocabulary size:', self.vocab_size)
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.idx2char[-1] = '-1'
        self.x = np.array([self.char2idx[char] for char in list(text)])
    # end method


    def next_batch(self, batch_size, text_iter_step):
        self.x_dropped = self._word_dropout(self.x, self.word_dropout_rate, self.char2idx)
        window = self.seq_len * batch_size
        for i in range(0, len(self.x)-window-1, text_iter_step):
            yield (self.x[i:i+window].reshape(-1, self.seq_len),
                   self.x_dropped[i:i+window].reshape(-1, self.seq_len))
    # end method


    def fit(self, text_iter_step=25, n_epoch=1, batch_size=128, save_mode=False):
        n_batch = (len(self.x) - self.seq_len*batch_size - 1) // text_iter_step
        self.sess.run(tf.global_variables_initializer()) # initialize all variables
        
        if save_mode:
            if os.path.isfile(self.model_path+'.meta'):
                print("Loading trained model ...")
                self.saver.restore(self.sess, self.model_path)
        
        for epoch in range(n_epoch):
            for local_step, (seq_batch, dropped_batch) in enumerate(self.next_batch(batch_size, text_iter_step)):
                _, nll_loss, kl_loss, kl_w, mutinfo_loss = self.sess.run(
                    [self.train_op, self.nll_loss, self.kl_loss, self.kl_w, self.mutinfo_loss],
                    {self.sequence: seq_batch,
                     self.sequence_dropped: dropped_batch,
                     self.sequence_length: [self.seq_len]*len(seq_batch)})
                if local_step % 10 == 0:
                    print ('Epoch %d/%d | Batch %d/%d | nll_loss: %.3f | kl_w: %.3f | kl_loss:%.3f | mutinfo_loss:%.3f' %
                          (epoch+1, n_epoch, local_step, n_batch, nll_loss, kl_w, kl_loss, mutinfo_loss), end='\n\n')
                if local_step % 100 == 0:
                    if save_mode:
                        save_path = self.saver.save(self.sess, self.model_path)
                        print("Model saved in file: %s" % save_path, end='\n\n')
                    
                    self.reconstruct(seq_batch[-1])
                    self.generate()
    # end method


    def generate(self):
        predicted_ids = self.sess.run(self.predicted_ids,
                                     {self._batch_size: 1,
                                      self._latent_vec: np.random.randn(1, self.latent_size),
                                      self.gen_seq_length: self.seq_len * 2})[0]
        print('G: '+''.join([self.idx2char[idx] for idx in predicted_ids]), end='\n\n')
    # end method


    def reconstruct(self, sentence):
        print("I: %s" % ''.join([self.idx2char[idx] for idx in sentence]), end='\n\n')
        predicted_ids = self.sess.run(self.predicted_ids,
            {self.sequence: np.atleast_2d(sentence),
             self.sequence_length: np.atleast_1d(len(sentence)),
             self.gen_seq_length: len(sentence) * 2})[0]
        print("O: %s" % ''.join([self.idx2char[idx] for idx in predicted_ids]), end='\n\n')
    # end method


    def _decoder_output(self):
        _end = tf.fill([self._batch_size, 1], self.char2idx['<end>'])
        return tf.concat([self.sequence, _end], 1)
    # end method


    def _decoder_input(self):
        _start = tf.fill([self._batch_size, 1], self.char2idx['<start>'])
        return tf.concat([_start, self.sequence_dropped], 1)
    # end method


    def _rnn_cell(self, reuse=False):
        return tf.nn.rnn_cell.GRUCell(self.rnn_size, kernel_initializer=tf.orthogonal_initializer(),
                                      reuse=reuse)
    # end method


    def _word_dropout(self, x, dropout_rate, word2idx):
        is_dropped = np.random.binomial(1, dropout_rate, x.shape)
        x_dropped = x.copy()
        for i in range(x.shape[0]):
            if is_dropped[i]:
                x_dropped[i] = word2idx['<unk>']
        return x_dropped
    # end method


    def _mutinfo_loss(self, z, z_mean, z_logvar):
        z = tf.stop_gradient(z)

        z_var = tf.exp(z_logvar) + 1e-8 # adjust for epsilon
        z_logvar = tf.log(z_var)
        
        z_sq = tf.square(z)
        z_epsilon = tf.square(z - z_mean)
        return 0.5 * tf.reduce_sum(z_logvar + (z_epsilon / z_var) - z_sq, 1)
    # end method
# end class
