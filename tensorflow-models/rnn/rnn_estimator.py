import tensorflow as tf
import numpy as np
tf.logging.set_verbosity(tf.logging.INFO)


class Estimator:
    def __init__(self, n_out, rnn_size=128, n_layers=2, dropout_rate=0.2):
        self.n_out = n_out
        self.rnn_size = rnn_size
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.model = tf.estimator.Estimator(self.model_fn)
    # end constructor


    def rnn_net(self, x_dict, reuse, dropout_rate):
        with tf.variable_scope('rnn_net', reuse=reuse):
            x = x_dict['sequences']
            seq_len = x_dict['sequence_lengths']

            cells = tf.nn.rnn_cell.MultiRNNCell([self.cell_fn(dropout_rate) for _ in range(self.n_layers)])
            _, final_state = tf.nn.dynamic_rnn(cells, x, sequence_length=seq_len, dtype=tf.float32)
            logits = tf.layers.dense(final_state[-1].h, self.n_out)
        return logits
    # end method


    def cell_fn(self, dropout_rate):
        cell = tf.nn.rnn_cell.LSTMCell(self.rnn_size, initializer=tf.orthogonal_initializer())
        cell = tf.nn.rnn_cell.DropoutWrapper(
            cell, state_keep_prob=1-dropout_rate)
        return cell
    # end method


    def model_fn(self, features, labels, mode):
        logits = self.rnn_net(features, reuse=False, dropout_rate=self.dropout_rate)
        predictions = tf.argmax(self.rnn_net(features, reuse=True, dropout_rate=0.0), axis=1)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))
        train_op = tf.train.AdamOptimizer().minimize(loss_op, global_step=tf.train.get_global_step())
        acc_op = tf.metrics.accuracy(labels=labels, predictions=predictions)

        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={'accuracy': acc_op})
        return estim_specs
    # end method


    def fit(self, x, y, batch_size=128, n_epoch=5):
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'sequences':x, 'sequence_lengths':np.array([28]*len(x))}, y=y,
            batch_size=batch_size, num_epochs=n_epoch, shuffle=True)
        self.model.train(input_fn)
    # end method


    def score(self, x_test, y_test, batch_size=128):
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'sequences': x_test, 'sequence_lengths':np.array([28]*len(x_test))}, y=y_test,
            batch_size=batch_size, shuffle=False)
        self.model.evaluate(input_fn)
    # end method