import tensorflow as tf
import numpy as np
from config import args
from tensorflow.python.layers import core as core_layers


class VRAE:
    def __init__(self, word2idx):
        self._word2idx = word2idx
        self._idx2word = {i: w for w, i in word2idx.items()}
        self._exception_handling()
        self._build_graph()

    
    def _build_graph(self):
        self._build_forward_graph()
        self._build_backward_graph()


    def _build_forward_graph(self):
        self._build_inputs()
        self._decoder(self._latent(self._encoder()))


    def _build_backward_graph(self):
        global_step = tf.Variable(0, trainable=False)

        self.nll_loss = self._nll_loss_fn()
        self.kl_w = self._kl_w_fn(args.anneal_max, args.anneal_bias, global_step)
        self.kl_loss = self._kl_loss_fn(self._mean, self._gamma)
        self.mutinfo_loss = self._mutinfo_loss_fn(self.latent_vec, self._mean, self._gamma)
        loss_op = self.nll_loss + self.kl_w * (self.kl_loss + args.mutinfo_weight * self.mutinfo_loss)

        params = tf.trainable_variables()
        gradients = tf.gradients(loss_op, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, args.clip_norm)
        self.train_op = tf.train.AdamOptimizer().apply_gradients(
            zip(clipped_gradients, params), global_step=global_step)


    def _build_inputs(self):
        # placeholders
        self.seq = tf.placeholder(tf.int32, [None, None])
        self.seq_dropped = tf.placeholder(tf.int32, [None, None])
        self.seq_length = tf.placeholder(tf.int32, [None])
        self.gen_seq_length = tf.placeholder(tf.int32, [])
        # dynmaic batch size
        self._batch_size = tf.shape(self.seq)[0]
        
    
    def _encoder(self):
        with tf.variable_scope('encoder'):
            encoded_output, encoded_state = tf.nn.dynamic_rnn(
                cell = tf.nn.rnn_cell.MultiRNNCell(
                    [self._residual_rnn_cell() for _ in range(args.encoder_layers)]), 
                inputs = tf.contrib.layers.embed_sequence(self.seq, args.vocab_size, args.embedding_dim),
                sequence_length = self.seq_length,
                dtype = tf.float32)
            return self._weighted_sum_encoded_output(encoded_output, encoded_state)


    def _latent(self, rnn_encoded):
        self._mean = tf.layers.dense(rnn_encoded, args.latent_size, tf.nn.elu)
        self._gamma = tf.layers.dense(rnn_encoded, args.latent_size, tf.nn.elu)
        _gaussian = tf.truncated_normal(tf.shape(self._gamma))

        self.latent_vec = self._mean + tf.exp(0.5 * self._gamma) * _gaussian
        if args.rnn_cell == 'lstm':
            c = tf.layers.dense(self.latent_vec, args.rnn_size, tf.nn.elu)
            h = tf.layers.dense(self.latent_vec, args.rnn_size, tf.nn.elu)
            return tuple([tf.nn.rnn_cell.LSTMStateTuple(c=c, h=h)] * args.decoder_layers)
        if args.rnn_cell == 'gru':
            state = tf.layers.dense(self.latent_vec, args.rnn_size, tf.nn.elu)
            return tuple([state] * args.decoder_layers)


    def _decoder(self, init_state):
        with tf.variable_scope('training'):
            self.training_rnn_out, self.training_logits = self._decoder_training(init_state)
        with tf.variable_scope('training', reuse=True):
            self.predicted_ids = self._decoder_inference(init_state)


    def _decoder_training(self, init_state):
        lin_proj = core_layers.Dense(args.vocab_size, _scope='decoder/dense')

        helper = tf.contrib.seq2seq.TrainingHelper(
            inputs = tf.contrib.layers.embed_sequence(
                self._decoder_input(), args.vocab_size, args.embedding_dim),
            sequence_length = self.seq_length + 1)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [self._residual_rnn_cell() for _ in range(args.decoder_layers)]),
            helper = helper,
            initial_state = init_state,
            output_layer = None)
        decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder = decoder,
            impute_finished = True,
            maximum_iterations = tf.reduce_max(self.seq_length + 1))
        
        return decoder_output.rnn_output, lin_proj.apply(decoder_output.rnn_output)


    def _decoder_inference(self, init_state):
        decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [self._residual_rnn_cell(reuse=True) for _ in range(args.decoder_layers)]),
            embedding = tf.get_variable('EmbedSequence/embeddings'),
            start_tokens = tf.tile(tf.constant(
                [self._word2idx['<start>']], dtype=tf.int32), [self._batch_size]),
            end_token = self._word2idx['<end>'],
            initial_state = tf.contrib.seq2seq.tile_batch(init_state, args.beam_width),
            beam_width = args.beam_width,
            output_layer = core_layers.Dense(args.vocab_size, _reuse=True),
            length_penalty_weight = 0.0)
        decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder = decoder,
            impute_finished = False,
            maximum_iterations = self.gen_seq_length)
        return decoder_output.predicted_ids[:, :, 0]


    def _rnn_cell(self, reuse=False):
        if args.rnn_cell == 'lstm':
            return tf.nn.rnn_cell.LSTMCell(
                args.rnn_size, initializer=tf.orthogonal_initializer(), reuse=reuse)
        if args.rnn_cell == 'gru':
            return tf.nn.rnn_cell.GRUCell(
                args.rnn_size, kernel_initializer=tf.orthogonal_initializer(), reuse=reuse)


    def _residual_rnn_cell(self, reuse=False):
        if args.embedding_dim != args.rnn_size:
            return tf.nn.rnn_cell.ResidualWrapper(
                tf.contrib.rnn.OutputProjectionWrapper(
                    self._rnn_cell(reuse=reuse), args.embedding_dim))
        if args.embedding_dim == args.rnn_size:
            return tf.nn.rnn_cell.ResidualWrapper(self._rnn_cell(reuse=reuse))


    def _decoder_input(self):
        _start = tf.fill([self._batch_size, 1], self._word2idx['<start>'])
        return tf.concat([_start, self.seq_dropped], 1)


    def _decoder_output(self):
        _end = tf.fill([self._batch_size, 1], self._word2idx['<end>'])
        return tf.concat([self.seq, _end], 1)


    def _nll_loss_fn(self):
        if args.num_sampled >= args.vocab_size:
            seq_length = self.seq_length + 1
            return tf.reduce_sum(tf.contrib.seq2seq.sequence_loss(
                logits = self.training_logits,
                targets = self._decoder_output(),
                weights = tf.sequence_mask(seq_length, tf.reduce_max(seq_length), dtype=tf.float32),
                average_across_timesteps = False,
                average_across_batch = True))
        if args.num_sampled < args.vocab_size:
            with tf.variable_scope('training/decoder/dense', reuse=True):
                return tf.reduce_sum(tf.nn.sampled_softmax_loss(
                    weights = tf.transpose(tf.get_variable('kernel')),
                    biases = tf.get_variable('bias'),
                    labels = tf.reshape(self._decoder_output(), [-1, 1]),
                    inputs = tf.reshape(self.training_rnn_out, [-1, args.rnn_size]),
                    num_sampled = args.num_sampled,
                    num_classes = args.vocab_size,
                )) / tf.cast(self._batch_size, tf.float32)


    def _kl_w_fn(self, anneal_max, anneal_bias, global_step):
        return anneal_max * tf.sigmoid((10 / anneal_bias) * (
            tf.cast(global_step, tf.float32) - tf.constant(anneal_bias / 2)))


    def _kl_loss_fn(self, mean, gamma):
        return 0.5 * tf.reduce_sum(
            tf.exp(gamma) + tf.square(mean) - 1 - gamma) / tf.cast(self._batch_size, tf.float32)


    def _mutinfo_loss_fn(self, z, z_mean, z_logvar):
        z = tf.stop_gradient(z)
        z_var = tf.exp(z_logvar) + 1e-8 # adjust for epsilon
        z_logvar = tf.log(z_var)
        z_sq = tf.square(z)
        z_epsilon = tf.square(z - z_mean)
        return tf.reduce_sum(
            0.5 * tf.reduce_sum(z_logvar + (z_epsilon / z_var) - z_sq, 1)) \
                / tf.cast(self._batch_size, tf.float32)


    def train_session(self, sess, seq, seq_dropped, seq_len):
        _, nll_loss, kl_w, kl_loss, mutinfo_loss = sess.run(
            [self.train_op, self.nll_loss, self.kl_w, self.kl_loss, self.mutinfo_loss],
                {self.seq: seq, self.seq_dropped: seq_dropped, self.seq_length: seq_len})
        return {'nll_loss': nll_loss,
                'kl_w': kl_w,
                'kl_loss': kl_loss,
                'mutinfo_loss': mutinfo_loss}


    def reconstruct(self, sess, sentence):
        print('\nI: %s' % ' '.join([self._idx2word[idx] for idx in sentence]), end='\n\n')
        predicted_ids = sess.run(self.predicted_ids,
            {self.seq: np.atleast_2d(sentence),
             self.seq_length: np.atleast_1d(len(sentence)),
             self.gen_seq_length: len(sentence)})[0]
        print('O: %s' % ' '.join([self._idx2word[idx] for idx in predicted_ids]), end='\n\n')


    def generate(self, sess):
        predicted_ids = sess.run(self.predicted_ids,
                                {self._batch_size: 1,
                                 self.latent_vec: np.random.randn(1, args.latent_size),
                                 self.gen_seq_length: args.max_len})[0]
        print('G: %s' % ' '.join([self._idx2word[idx] for idx in predicted_ids]), end='\n\n')


    def _exception_handling(self):
        self._idx2word[-1] = '-1'
        self._idx2word[4] = '4'


    def _weighted_sum_encoded_output(self, encoded_output, encoded_state):
        if args.rnn_cell == 'lstm':
            encoded_state = encoded_state[-1].h
        if args.rnn_cell == 'gru':
            encoded_state = encoded_state[-1]
        
        weights = tf.nn.softmax(tf.squeeze(tf.matmul(
            encoded_output, tf.expand_dims(encoded_state,2)), 2))
        weighted_sum = tf.squeeze(tf.matmul(
            tf.transpose(encoded_output,[0,2,1]), tf.expand_dims(weights,2)), 2)
        return weighted_sum
