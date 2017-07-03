import tensorflow as tf
from tensorflow.python.layers.core import Dense
import numpy as np


class Seq2Seq:
    def __init__(self, rnn_size, n_layers,
                 X_word2idx, encoder_embedding_dim,
                 Y_word2idx, decoder_embedding_dim,
                 batch_size, sess=tf.Session(),
                 ):
        self.rnn_size = rnn_size
        self.n_layers = n_layers
        self.X_word2idx = X_word2idx
        self.encoder_embedding_dim = encoder_embedding_dim
        self.Y_word2idx = Y_word2idx
        self.decoder_embedding_dim = decoder_embedding_dim
        self.batch_size = batch_size
        self.sess = sess
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.add_input_layer()
        self.add_encoder_layer()
        self.add_decoder_layer()
        self.add_backward_path()
    # end method build_graph


    def add_input_layer(self):
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, None])
        self.X_seq_len = tf.placeholder(tf.int32, [None, ])
        self.Y_seq_len = tf.placeholder(tf.int32, [None, ])
    # end method add_input_layer


    def lstm_cell(self):
        return tf.nn.rnn_cell.LSTMCell(self.rnn_size, initializer=tf.orthogonal_initializer)
    # end method lstm_cell


    def add_encoder_layer(self):            
        _, self.encoder_state = tf.nn.dynamic_rnn(
            tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell() for _ in range(self.n_layers)]), 
            tf.contrib.layers.embed_sequence(self.X, len(self.X_word2idx), self.encoder_embedding_dim),
            sequence_length = self.X_seq_len,
            dtype = tf.float32,
        )
    # end method add_encoder_layer
    

    def process_decoder_input(self, data, word2idx, batch_size):
        ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1]) # remove last char
        decoder_input = tf.concat([tf.fill([batch_size, 1], word2idx['<GO>']), ending], 1)
        return decoder_input
    # end method add_decoder_layer


    def add_decoder_layer(self):
        decoder_cell = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell() for _ in range(self.n_layers)])
        decoder_input = self.process_decoder_input(self.Y, self.Y_word2idx, self.batch_size)
        Y_vocab_size = len(self.Y_word2idx)
        decoder_embedding = tf.get_variable('decoder_embedding',
                                            [Y_vocab_size, self.decoder_embedding_dim],
                                            tf.float32,
                                            tf.random_uniform_initializer(-1.0, 1.0))
        output_layer = Dense(Y_vocab_size, kernel_initializer = tf.truncated_normal_initializer(stddev=0.1))
        self.max_Y_seq_len = tf.reduce_max(self.Y_seq_len)

        with tf.variable_scope('decode'):
            training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs = tf.nn.embedding_lookup(decoder_embedding, decoder_input),
                sequence_length = self.Y_seq_len,
                time_major = False)
            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell = decoder_cell,
                helper = training_helper,
                initial_state = self.encoder_state,
                output_layer = output_layer)
            training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = training_decoder,
                impute_finished = True,
                maximum_iterations = self.max_Y_seq_len)
            self.training_logits = training_decoder_output.rnn_output
        
        with tf.variable_scope('decode', reuse=True):
            start_tokens = tf.tile(tf.constant([self.Y_word2idx['<GO>']], dtype=tf.int32), [self.batch_size])
            predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding = decoder_embedding,
                start_tokens = start_tokens,
                end_token = self.Y_word2idx['<EOS>'])
            predicting_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell = decoder_cell,
                helper = predicting_helper,
                initial_state = self.encoder_state,
                output_layer = output_layer)
            predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = predicting_decoder,
                impute_finished = True,
                maximum_iterations = self.max_Y_seq_len)
            self.predicting_logits = predicting_decoder_output.sample_id
    # end method add_decoder_layer


    def add_backward_path(self):
        masks = tf.sequence_mask(self.Y_seq_len, self.max_Y_seq_len, dtype=tf.float32)

        self.loss = tf.contrib.seq2seq.sequence_loss(logits = self.training_logits,
                                                     targets = self.Y,
                                                     weights = masks)
        
        optimizer = tf.train.AdamOptimizer()
        gradients = optimizer.compute_gradients(self.loss)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        self.train_op = optimizer.apply_gradients(capped_gradients)
    # end method add_backward_path


    def pad_sentence_batch(self, sentence_batch, pad_int):
        max_sentence_len = max([len(sentence) for sentence in sentence_batch])
        pad_sentence_batch = []
        pad_length_batch = []

        for sentence in sentence_batch:
            pad_sentence_batch.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
            pad_length_batch.append(max_sentence_len)

        return pad_sentence_batch, pad_length_batch
    # end method pad_sentence_batch


    def gen_batch(self, sources, targets, batch_size, source_pad_int=None, target_pad_int=None):
        if source_pad_int is None:
            source_pad_int = self.X_word2idx['<PAD>']
        if target_pad_int is None:
            target_pad_int = self.Y_word2idx['<PAD>']
        
        for batch_i in range(0, len(sources)//batch_size):
            start_i = batch_i * batch_size
            sources_batch = sources[start_i : start_i + batch_size]
            targets_batch = targets[start_i : start_i + batch_size]
            
            pad_sources_batch, pad_sources_len = self.pad_sentence_batch(sources_batch, source_pad_int)
            pad_targets_batch, pad_targets_len = self.pad_sentence_batch(targets_batch, target_pad_int)
            
            yield np.array(pad_sources_batch), np.array(pad_targets_batch), pad_sources_len, pad_targets_len
    # end method gen_batch


    def fit(self, X_train, Y_train, val_data, n_epoch=60, display_step=50):
        X_test, Y_test = val_data
        X_test_batch, Y_test_batch, X_test_batch_lens, Y_test_batch_lens = next(self.gen_batch(X_test, Y_test, self.batch_size))

        self.sess.run(tf.global_variables_initializer())
        for epoch in range(1, n_epoch+1):
            for local_step, (X_train_batch, Y_train_batch, X_train_batch_lens, Y_train_batch_lens) in enumerate(
                self.gen_batch(X_train, Y_train, self.batch_size)):
                _, loss = self.sess.run([self.train_op, self.loss], {self.X: X_train_batch,
                                                                     self.Y: Y_train_batch,
                                                                     self.X_seq_len: X_train_batch_lens,
                                                                     self.Y_seq_len: Y_train_batch_lens})
                if local_step % display_step == 0:
                    val_loss = self.sess.run(self.loss, {self.X: X_test_batch,
                                                        self.Y: Y_test_batch,
                                                        self.X_seq_len: X_test_batch_lens,
                                                        self.Y_seq_len: Y_test_batch_lens})
                    print("Epoch %d/%d | Batch %d/%d | Train Loss %.3f | Test Loss %.3f"
                        % (epoch, n_epoch, local_step, len(X_train)//self.batch_size, loss, val_loss))
    # end method fit

    def infer(self, input_word, X_idx2char, Y_idx2char):
        def preprocess(word):
            seq_len = len(word) + 1
            return [self.X_word2idx.get(char, self.X_word2idx['<UNK>']) for char in word] + [pad]*(seq_len-len(word))
        
        pad = self.X_word2idx['<PAD>']
        indexed = preprocess(input_word)
        logits = self.sess.run(self.predicting_logits, {self.X: [indexed] * self.batch_size,
                                                                self.X_seq_len: [len(indexed)] * self.batch_size,
                                                                self.Y_seq_len: [len(indexed)] * self.batch_size})[0]
        
        print('Source')
        print('Word: {}'.format([i for i in indexed]))
        print('Input Words: {}'.format(' '.join([X_idx2char[i] for i in indexed])))
        
        print('\nTarget')
        print('Word: {}'.format([i for i in logits if i != pad]))
        print('Response Words: {}'.format(' '.join([Y_idx2char[i] for i in logits if i != pad])))
    # end method infer
# end class