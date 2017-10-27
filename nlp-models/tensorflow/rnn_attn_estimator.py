import tensorflow as tf
import numpy as np
tf.logging.set_verbosity(tf.logging.INFO)


class Estimator:
    def __init__(self, vocab_size, n_out, embedding_dims=128, rnn_size=128, dropout_rate=0.2, grad_clip=5.0,
                 attn_size=50):
        self.vocab_size = vocab_size
        self.n_out = n_out
        self.embedding_dims = embedding_dims
        self.rnn_size = rnn_size
        self.dropout_rate = dropout_rate
        self.grad_clip = grad_clip
        self.attn_size = attn_size
        self.model = tf.estimator.Estimator(self.model_fn)
    # end constructor


    def rnn_net(self, x_dict, reuse, dropout_rate):
        with tf.variable_scope('rnn_net', reuse=reuse):
            x = x_dict['sequences']
            seq_len = x_dict['sequence_lengths']

            embedding = tf.get_variable(
                'encoder', [self.vocab_size, self.embedding_dims], tf.float32,
                tf.random_uniform_initializer(-1.0, 1.0))
            embedded = tf.nn.dropout(tf.nn.embedding_lookup(embedding, x), 1-dropout_rate)

            cell = tf.nn.rnn_cell.LSTMCell(self.rnn_size, initializer=tf.orthogonal_initializer())
            rnn_out, final_state = tf.nn.dynamic_rnn(cell, embedded, sequence_length=seq_len, dtype=tf.float32)

            weights = tf.layers.dense(tf.layers.dense(rnn_out, self.attn_size, tf.tanh), 1)
            weights = self._softmax(tf.squeeze(weights, 2))
            weighted_sum = tf.squeeze(tf.matmul(
                tf.transpose(rnn_out, [0, 2, 1]), tf.expand_dims(weights, 2)), 2)

            logits = tf.layers.dense(tf.concat([weighted_sum, final_state.h], 1), self.n_out)
        return logits
    # end method


    def model_fn(self, features, labels, mode):
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.PREDICT:
            logits = self.rnn_net(features, reuse=False, dropout_rate=self.dropout_rate)
            predictions = tf.argmax(self.rnn_net(features, reuse=True, dropout_rate=0.0), axis=1)
        
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        if mode == tf.estimator.ModeKeys.TRAIN:
            loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels))
            acc_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1), labels), tf.float32))
            acc_hook = tf.train.LoggingTensorHook({'acc':acc_op}, every_n_iter=100)

            params = tf.trainable_variables()
            gradients = tf.gradients(loss_op, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
            train_op = tf.train.AdamOptimizer().apply_gradients(
                zip(clipped_gradients, params), global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss_op,
                train_op=train_op,
                training_hooks = [acc_hook])
    # end method


    def fit(self, x, x_seq_len, y, batch_size=128, n_epoch=10):
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'sequences':x, 'sequence_lengths':x_seq_len}, y=y,
            batch_size=batch_size, num_epochs=n_epoch, shuffle=True)
        self.model.train(input_fn)
    # end method


    def predict(self, x_test, x_seq_len, batch_size=128):
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'sequences': x_test, 'sequence_lengths':x_seq_len},
            batch_size=batch_size, shuffle=False)
        return np.array(list(self.model.predict(input_fn)))
    # end method


    def _softmax(self, tensor):
        exps = tf.exp(tensor)
        return exps / tf.reduce_sum(exps, 1, keep_dims=True)
    # end method softmax
# end class