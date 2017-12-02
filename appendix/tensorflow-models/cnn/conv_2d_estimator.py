import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)


class Estimator:
    def __init__(self, n_out, dropout_rate=0.5):
        self.n_out = n_out
        self.dropout_rate = dropout_rate
        self.model = tf.estimator.Estimator(self.model_fn)
    # end constructor


    def conv_net(self, x_dict, reuse, is_training):
        with tf.variable_scope('conv_net', reuse=reuse):
            x = x_dict['images']

            conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
            conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

            conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
            conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

            fc = tf.contrib.layers.flatten(conv2)
            fc = tf.layers.dense(fc, 1024)
            fc = tf.layers.dropout(fc, rate=self.dropout_rate, training=is_training)
            out = tf.layers.dense(fc, self.n_out)
        return out
    # end method


    def model_fn(self, features, labels, mode):
        logits = self.conv_net(features, reuse=False, is_training=True)
        predictions = tf.argmax(self.conv_net(features, reuse=True, is_training=False), axis=1)

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


    def fit(self, x, y, batch_size=128, n_epoch=1):
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images':x}, y=y, batch_size=batch_size, num_epochs=n_epoch, shuffle=True)
        self.model.train(input_fn)
    # end method


    def score(self, x_test, y_test, batch_size=128):
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': x_test}, y=y_test, batch_size=batch_size, shuffle=False)
        self.model.evaluate(input_fn)
    # end method