import tensorflow as tf
import numpy as np
from config import args
from tensorflow.python.layers.core import Dense
# we modify the source classes to make sure we can concat Z into every decoder input stage
from modified_tf_classes import BasicDecoder, BeamSearchDecoder


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
        self._decoder_to_output(
            self._latent_to_decoder(
                self._encoder_to_latent()))
        if args.mutinfo_loss:
            self._output_to_latent()


    def _build_backward_graph(self):
        if not args.mutinfo_loss:
            self.global_step = tf.Variable(0, trainable=False)

        self.nll_loss = self._nll_loss_fn()
        self.kl_w = self._kl_w_fn(args.anneal_max, args.anneal_bias, self.global_step)
        self.kl_loss = self._kl_loss_fn(self.z_mean, self.z_logvar)
        
        loss_op = self.nll_loss + self.kl_w * self.kl_loss

        if args.mutinfo_loss:
            self.mutinfo_loss = self._mutinfo_loss_fn(self.z_mean_new, self.z_logvar_new)
            loss_op += self.mutinfo_loss
        
        clipped_gradients, params = self._gradient_clipping(loss_op)
        self.train_op = tf.train.AdamOptimizer().apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step)


    def _build_inputs(self):
        # placeholders
        self.seq = tf.placeholder(tf.int32, [None, None])
        self.seq_dropped = tf.placeholder(tf.int32, [None, None])
        self.seq_length = tf.placeholder(tf.int32, [None])
        self.gen_seq_length = tf.placeholder(tf.int32, [])
        # dynmaic batch size
        self._batch_size = tf.shape(self.seq)[0]
        
    
    def _encoder_to_latent(self):
        # the embedding is shared between encoder and decoder
        # since the source and the target for an autoencoder are the same
        self.tied_embedding = tf.get_variable('tied_embedding', [args.vocab_size, args.embedding_dim],
            tf.float32, tf.random_uniform_initializer(-1.0, 1.0))
                
        with tf.variable_scope('encoder'):
            encoded_output, encoded_state = tf.nn.dynamic_rnn(
                cell = tf.nn.rnn_cell.MultiRNNCell(
                    [self._residual_rnn_cell() for _ in range(args.encoder_layers)]), 
                inputs = tf.nn.embedding_lookup(self.tied_embedding, self.seq),
                sequence_length = self.seq_length,
                dtype = tf.float32)
        return self._parse_encoded_state(encoded_state)


    def _latent_to_decoder(self, rnn_encoded):
        with tf.variable_scope('latent'):
            self.z_mean = tf.layers.dense(rnn_encoded, args.latent_size)
            self.z_logvar = tf.layers.dense(rnn_encoded, args.latent_size, tf.nn.softplus)
            self.gaussian_noise = tf.truncated_normal(tf.shape(self.z_logvar))
            self.z = self.z_mean + tf.exp(0.5 * self.z_logvar) * self.gaussian_noise
            self.z_gated = self.z
            for _ in range(2):
                self.z_gated = self._highway(self.z_gated)
            return self._parse_decoder_state(self.z_gated)


    def _decoder_to_output(self, init_state):
        with tf.variable_scope('decoding'):
            self.training_rnn_out, self.training_logits = self._decoder_training(init_state)
        with tf.variable_scope('decoding', reuse=True):
            self.predicted_ids = self._decoder_inference(init_state)


    def _decoder_training(self, init_state):
        lin_proj = Dense(args.vocab_size, _scope='decoder/dense')

        helper = tf.contrib.seq2seq.TrainingHelper(
            inputs = tf.nn.embedding_lookup(self.tied_embedding, self._decoder_input()),
            sequence_length = self.seq_length + 1)
        decoder = BasicDecoder(
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [self._rnn_cell() for _ in range(args.decoder_layers)]),
            helper = helper,
            initial_state = init_state,
            concat_z = self.z_gated,
            output_layer = None)
        decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder = decoder,
            impute_finished = True,
            maximum_iterations = tf.reduce_max(self.seq_length + 1))
        
        return decoder_output.rnn_output, lin_proj.apply(decoder_output.rnn_output)


    def _decoder_inference(self, init_state):
        tiled_z = tf.tile(tf.expand_dims(self.z_gated, 1), [1, args.beam_width, 1])

        decoder = BeamSearchDecoder(
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [self._rnn_cell(reuse=True) for _ in range(args.decoder_layers)]),
            embedding = self.tied_embedding,
            start_tokens = tf.tile(tf.constant(
                [self._word2idx['<start>']], dtype=tf.int32), [self._batch_size]),
            end_token = self._word2idx['<end>'],
            initial_state = tf.contrib.seq2seq.tile_batch(init_state, args.beam_width),
            beam_width = args.beam_width,
            output_layer = Dense(args.vocab_size, _reuse=True),
            length_penalty_weight = 0.0,
            concat_z = tiled_z)
        decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder = decoder,
            impute_finished = False,
            maximum_iterations = self.gen_seq_length)

        return decoder_output.predicted_ids[:, :, 0]


    def _rnn_cell(self, reuse=False):
        if args.rnn_cell == 'lstm':
            return tf.nn.rnn_cell.LSTMCell(
                args.rnn_size, initializer=tf.orthogonal_initializer(), reuse=reuse)
        elif args.rnn_cell == 'gru':
            return tf.nn.rnn_cell.GRUCell(
                args.rnn_size, kernel_initializer=tf.orthogonal_initializer(), reuse=reuse)
        else:
            raise ValueError("rnn_cell must be one of 'lstm' or 'gru'")


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
        seq_length = self.seq_length + 1
        self.mask = tf.sequence_mask(seq_length, tf.reduce_max(seq_length), dtype=tf.float32)
        if (args.num_sampled == 0) or (args.num_sampled >= args.vocab_size):
            return tf.reduce_sum(tf.contrib.seq2seq.sequence_loss(
                logits = self.training_logits,
                targets = self._decoder_output(),
                weights = self.mask,
                average_across_timesteps = False,
                average_across_batch = True))
        else:
            with tf.variable_scope('decoding/decoder/dense', reuse=True):
                return tf.reduce_sum(tf.reshape(self.mask,[-1]) * tf.nn.sampled_softmax_loss(
                    weights = tf.transpose(tf.get_variable('kernel')),
                    biases = tf.get_variable('bias'),
                    labels = tf.reshape(self._decoder_output(), [-1, 1]),
                    inputs = tf.reshape(self.training_rnn_out, [-1, args.rnn_size]),
                    num_sampled = args.num_sampled,
                    num_classes = args.vocab_size,
                )) / tf.to_float(self._batch_size)


    def _kl_w_fn(self, anneal_max, anneal_bias, global_step):
        return anneal_max * tf.sigmoid((10 / anneal_bias) * (
            tf.cast(global_step, tf.float32) - tf.constant(anneal_bias / 2)))


    def _kl_loss_fn(self, mean, gamma):
        return 0.5 * tf.reduce_sum(
            tf.exp(gamma) + tf.square(mean) - 1 - gamma) / tf.to_float(self._batch_size)


    def train_session(self, sess, seq, seq_dropped, seq_len):
        if args.mutinfo_loss:
            _, nll_loss, kl_w, kl_loss, temperature, step, mutinfo_loss = sess.run(
                [self.train_op, self.nll_loss, self.kl_w, self.kl_loss, self.temperature, self.global_step,
                self.mutinfo_loss],
                    {self.seq: seq, self.seq_dropped: seq_dropped, self.seq_length: seq_len})
            return {'nll_loss': nll_loss,
                    'kl_w': kl_w,
                    'kl_loss': kl_loss,
                    'temperature': temperature,
                    'step': step,
                    'mutinfo_loss': mutinfo_loss}
        else:
            _, nll_loss, kl_w, kl_loss, step = sess.run(
                [self.train_op, self.nll_loss, self.kl_w, self.kl_loss, self.global_step],
                    {self.seq: seq, self.seq_dropped: seq_dropped, self.seq_length: seq_len})
            return {'nll_loss': nll_loss,
                    'kl_w': kl_w,
                    'kl_loss': kl_loss,
                    'step': step}


    def reconstruct(self, sess, sentence, sentence_dropped):
        print('\nI: %s' % ' '.join([self._idx2word[idx] for idx in sentence]), end='\n\n')
        print('D: %s' % ' '.join([self._idx2word[idx] for idx in sentence_dropped]), end='\n\n')
        predicted_ids = sess.run(self.predicted_ids,
            {self.seq: np.atleast_2d(sentence),
             self.seq_length: np.atleast_1d(len(sentence)),
             self.gen_seq_length: len(sentence)})[0]
        print('O: %s' % ' '.join([self._idx2word[idx] for idx in predicted_ids]), end='\n\n')


    def generate(self, sess):
        predicted_ids = sess.run(self.predicted_ids,
                                {self._batch_size: 1,
                                 self.z: np.random.randn(1, args.latent_size),
                                 self.gen_seq_length: args.max_len})[0]
        print('G: %s' % ' '.join([self._idx2word[idx] for idx in predicted_ids]), end='\n\n')


    def _exception_handling(self):
        self._idx2word[-1] = '-1' # bug in beam search decoder
        self._idx2word[4] = '4'   # bug in IMDB dataset offered by Keras


    def _gradient_clipping(self, loss_op):
        params = tf.trainable_variables()
        gradients = tf.gradients(loss_op, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, args.clip_norm)
        return clipped_gradients, params


    def _parse_encoded_state(self, encoded_state):
        if args.rnn_cell == 'lstm':
            encoded_state = encoded_state[-1].h
        elif args.rnn_cell == 'gru':
            encoded_state = encoded_state[-1]
        else:
            raise ValueError("rnn_cell must be one of 'lstm' or 'gru'")
        return encoded_state


    def _inverse_sigmoid(self, x):
        return 1 / (1 + tf.exp(x))


    def _temperature_fn(self, anneal_max, anneal_bias, global_step):
        return anneal_max * self._inverse_sigmoid((10 / anneal_bias) * (
            tf.cast(global_step, tf.float32) - tf.constant(anneal_bias / 2)))


    def _output_to_latent(self):
        self.global_step = tf.Variable(0, trainable=False)
        self.temperature = self._temperature_fn(args.anneal_max, args.anneal_bias, self.global_step)

        gumble = tf.contrib.distributions.RelaxedOneHotCategorical(
            self.temperature, logits=self.training_logits)
        embeded_2d = tf.matmul(tf.reshape(gumble.sample(), [-1,args.vocab_size]), self.tied_embedding)
        embeded = tf.reshape(embeded_2d, [self._batch_size, args.max_len+1, args.embedding_dim])

        with tf.variable_scope('encoder', reuse=True):
            encoded_output, encoded_state = tf.nn.dynamic_rnn(
                cell = tf.nn.rnn_cell.MultiRNNCell(
                    [self._residual_rnn_cell(reuse=True) for _ in range(args.encoder_layers)]), 
                inputs = embeded,
                sequence_length = self.seq_length,
                dtype = tf.float32)
        encoded_state = self._parse_encoded_state(encoded_state)

        with tf.variable_scope('latent', reuse=True):
            self.z_mean_new = tf.layers.dense(encoded_state, args.latent_size, tf.nn.elu, reuse=True)
            self.z_logvar_new = tf.layers.dense(encoded_state, args.latent_size, tf.nn.elu, reuse=True)


    def _mutinfo_loss_fn(self, z_mean_new, z_logvar_new):
        epsilon = tf.constant(1e-10)
        distribution = tf.contrib.distributions.MultivariateNormalDiag(
            self.z_mean_new, tf.exp(self.z_logvar_new), validate_args=True)
        mutinfo_loss = -tf.log(tf.add(epsilon, distribution.prob(self.gaussian_noise)))
        return tf.reduce_sum(mutinfo_loss) / tf.to_float(self._batch_size)


    def _highway(self, x, carry_bias=-1.0, activation=tf.nn.elu):
            size = x.get_shape().as_list()[-1]
            H = tf.layers.dense(x, size, activation)
            T = tf.layers.dense(x, size, tf.sigmoid, bias_initializer=tf.constant_initializer(carry_bias))
            C = tf.subtract(1.0, T)
            y = tf.add(tf.multiply(H, T), tf.multiply(x, C)) # y = (H * T) + (x * C)
            return y
        # end add_highway


    def _parse_decoder_state(self, z):
        if args.rnn_cell == 'lstm':
            c = tf.layers.dense(self.z, args.rnn_size)
            h = tf.layers.dense(self.z, args.rnn_size)
            return tuple([tf.nn.rnn_cell.LSTMStateTuple(c=c, h=h)] * args.decoder_layers)
        elif args.rnn_cell == 'gru':
            state = tf.layers.dense(self.z, args.rnn_size)
            return tuple([state] * args.decoder_layers)
        else:
            raise ValueError("rnn_cell must be one of 'lstm' or 'gru'")
