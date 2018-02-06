#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import numpy as np
import os

def pad_sentence_batch(sentence_batch, pad_int):
    padded_seqs = []
    seq_lens = []
    max_sentence_len = max([len(sentence) for sentence in sentence_batch])
    for sentence in sentence_batch:
        padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
        seq_lens.append(len(sentence))
    return padded_seqs, seq_lens


class Seq2Seq:
    def __init__(self, rnn_size, n_layers, x_word2idx, encoder_embedding_dim, y_word2idx, decoder_embedding_dim,
                 x_embs=None, y_embs=None, sess=tf.Session(), grad_clip=5.0, model_path='./my_model'):
        self.rnn_size = rnn_size
        self.n_layers = n_layers
        self.grad_clip = grad_clip
        self.x_word2idx = x_word2idx
        self.encoder_embedding_dim = encoder_embedding_dim
        self.y_word2idx = y_word2idx
        self.decoder_embedding_dim = decoder_embedding_dim
        self.sess = sess
        self.model_path = model_path if model_path else 'my_model'
        self.saver = None
        self.predict_op = None

        model_dir = os.path.dirname(self.model_path)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        self._x_go = self.x_word2idx['<go>']
        self._x_eos = self.x_word2idx['<eos>']
        self._x_pad = self.x_word2idx['<pad>']
        self._x_unk = self.x_word2idx['<unk>']

        self._y_go = self.y_word2idx['<go>']
        self._y_eos = self.y_word2idx['<eos>']
        self._y_pad = self.y_word2idx['<pad>']
        self._y_unk = self.y_word2idx['<unk>']

    def restore_graph(self):
        # self.saver =
        self.saver = tf.train.import_meta_graph(self.model_path + '.meta')
        self.saver.restore(self.sess, tf.train.latest_checkpoint(os.path.dirname(self.model_path)))

    def build_graph(self):
        self.X = tf.placeholder(tf.int32, [None, None], 'X')
        self.Y = tf.placeholder(tf.int32, [None, None], 'Y')
        self.X_seq_len = tf.placeholder(tf.int32, [None], 'X_seq_len')
        self.Y_seq_len = tf.placeholder(tf.int32, [None], 'Y_seq_len')
        self.batch_size = tf.placeholder(tf.int32, [], 'batch_size')

        print self.X.name
        print self.Y.name
        print self.X_seq_len.name
        self.build_encoder_layer()
        self.build_decoder_layer()
        self.build_backward_path()

    def lstm_cell(self, reuse=False):
        return tf.nn.rnn_cell.LSTMCell(self.rnn_size, initializer=tf.orthogonal_initializer(), reuse=reuse)

    def build_encoder_layer(self):
        encoder_embedding = tf.get_variable('encoder_embedding', [len(self.x_word2idx), self.encoder_embedding_dim],
                                            tf.float32, tf.random_uniform_initializer(-1.0, 1.0))

        self.encoder_out, self.encoder_state = tf.nn.dynamic_rnn(
            cell=tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell() for _ in range(self.n_layers)]),
            inputs=tf.nn.embedding_lookup(encoder_embedding, self.X),
            sequence_length=self.X_seq_len,
            dtype=tf.float32)
        self.encoder_state = tuple(self.encoder_state[-1] for _ in range(self.n_layers))

    def _attention(self, reuse=False):
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units=self.rnn_size,
            memory=self.encoder_out,
            memory_sequence_length=self.X_seq_len)

        return tf.contrib.seq2seq.AttentionWrapper(
            cell=tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell(reuse) for _ in range(self.n_layers)]),
            attention_mechanism=attention_mechanism,
            attention_layer_size=self.rnn_size)

    def processed_decoder_input(self):
        main = tf.strided_slice(self.Y, [0, 0], [self.batch_size, -1], [1, 1]) # remove last char
        decoder_input = tf.concat([tf.fill([self.batch_size, 1], self._y_go), main], 1)
        return decoder_input
    # end method add_decoder_layer

    def build_decoder_layer(self):
        with tf.variable_scope('decode'):
            decoder_embedding = tf.get_variable('decoder_embedding', [len(self.y_word2idx), self.decoder_embedding_dim],
                                                tf.float32, tf.random_uniform_initializer(-1.0, 1.0))
            decoder_cell = self._attention()
            training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=tf.nn.embedding_lookup(decoder_embedding, self.processed_decoder_input()),
                sequence_length=self.Y_seq_len,
                time_major=False)
            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell,
                helper=training_helper,
                initial_state=decoder_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=self.encoder_state),
                output_layer=tf.layers.Dense(len(self.y_word2idx)))
            training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=training_decoder,
                impute_finished=True,
                maximum_iterations=tf.reduce_max(self.Y_seq_len))
            self.training_logits = training_decoder_output.rnn_output
        
        with tf.variable_scope('decode', reuse=True):
            decoder_cell = self._attention(reuse=True)
            predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=tf.get_variable('decoder_embedding'),
                start_tokens=tf.tile(tf.constant([self._y_go], dtype=tf.int32), [self.batch_size]),
                end_token=self._y_eos)
            predicting_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell,
                helper=predicting_helper,
                initial_state=decoder_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=self.encoder_state),
                output_layer=tf.layers.Dense(len(self.y_word2idx), _reuse=True))
            predicting_decoder_output, _, _=tf.contrib.seq2seq.dynamic_decode(
                decoder=predicting_decoder,
                impute_finished=True,
                maximum_iterations=2 * tf.reduce_max(self.X_seq_len))
            self.predicting_ids = predicting_decoder_output.sample_id
            print self.predicting_ids.name

    def build_backward_path(self):
        masks = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype=tf.float32)
        self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.training_logits, targets=self.Y, weights=masks)

        # gradient clipping
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
        self.train_op = tf.train.AdamOptimizer().apply_gradients(zip(clipped_gradients, params))

    def next_batch(self, x, y, batch_size):
        for i in range(0, len(x) - len(x) % batch_size, batch_size):
            x_batch = x[i: i + batch_size]
            y_batch = y[i: i + batch_size]
            padded_x_batch, x_batch_lens = pad_sentence_batch(x_batch, self._x_pad)
            padded__y_batch, y_batch_lens = pad_sentence_batch(y_batch, self._y_pad)
            yield (np.array(padded_x_batch),
                   np.array(padded__y_batch),
                   x_batch_lens,
                   y_batch_lens)

    def fit(self, x_train, y_train, val_data, n_epoch=10, display_step=500, batch_size=128):
        x_test, y_test = val_data
        x_test_batch, y_test_batch, x_test_batch_lens, y_test_batch_lens = \
            next(self.next_batch(x_test, y_test, batch_size))

        self.sess.run(tf.global_variables_initializer())

        best_metric_val = float('inf')
        self.saver = tf.train.Saver()

        for epoch in range(1, n_epoch+1):
            for local_step, (X_train_batch, Y_train_batch, X_train_batch_lens, Y_train_batch_lens) \
                    in enumerate(self.next_batch(x_train, y_train, batch_size)):
                _, loss = self.sess.run([self.train_op, self.loss], {self.X: X_train_batch,
                                                                     self.Y: Y_train_batch,
                                                                     self.X_seq_len: X_train_batch_lens,
                                                                     self.Y_seq_len: Y_train_batch_lens,
                                                                     self.batch_size: batch_size})
                if local_step % display_step == 0:
                    val_loss = self.sess.run(self.loss, {self.X: x_test_batch,
                                                         self.Y: y_test_batch,
                                                         self.X_seq_len: x_test_batch_lens,
                                                         self.Y_seq_len: y_test_batch_lens,
                                                         self.batch_size: batch_size})
                    print("Epoch %d/%d | Batch %d/%d | train_loss: %.3f | test_loss: %.3f"
                          % (epoch, n_epoch, local_step, len(x_train) // batch_size, loss, val_loss))

            val_loss = self.sess.run(self.loss, {self.X: x_test_batch,
                                                 self.Y: y_test_batch,
                                                 self.X_seq_len: x_test_batch_lens,
                                                 self.Y_seq_len: y_test_batch_lens,
                                                 self.batch_size: batch_size})

            print("Epoch %d/%d | test_loss: %.3f"
                  % (epoch, n_epoch, val_loss))
            if val_loss < best_metric_val:
                print 'Storing the model'
                self.saver.save(self.sess, self.model_path)  # self.model_dir)
                best_metric_val = val_loss

    def infer_sentence(self, input_word, x_idx2word, y_idx2word, batch_size=128):

        graph = tf.get_default_graph()
        predict_op = graph.get_tensor_by_name("decode_1/decoder/transpose_1:0")
        x_tensor = graph.get_tensor_by_name('X:0')
        x_len_tensor = graph.get_tensor_by_name('X_seq_len:0')
        batch_size_tensor = graph.get_tensor_by_name('batch_size:0')

        input_indices = [self.x_word2idx.get(char, self._x_unk) for char in input_word.strip().split()]
        out_indices = self.sess.run(predict_op, {
            x_tensor: [input_indices] * batch_size,
            x_len_tensor: [len(input_indices)] * batch_size,
            batch_size_tensor: batch_size})[0]
        
        print('\nSource')
        print('Word: {}'.format([i for i in input_indices]))
        out_str = 'IN: {}'.format(' '.join([x_idx2word[i].encode('utf8') for i in input_indices]))
        print(out_str)
        
        print('\nTarget')
        print('Word: {}'.format([i for i in out_indices]))
        print('OUT: {}'.format(' '.join([y_idx2word[i] for i in out_indices])))
