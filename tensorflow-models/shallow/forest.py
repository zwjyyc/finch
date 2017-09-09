from tensorflow.contrib.tensor_forest.python import tensor_forest
import tensorflow as tf
import numpy as np


class Forest:
    def __init__(self, n_features, n_classes, n_trees=10, max_nodes=1000, sess=tf.Session()):
        self.n_features = n_features
        self.hparams = tensor_forest.ForestHParams(num_classes = n_classes,
                                                   num_features = n_features,
                                                   num_trees = n_trees,
                                                   max_nodes = max_nodes).fill()
        self.sess = sess
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.n_features])
        self.y = tf.placeholder(tf.int32, shape=[None])
        forest_graph = tensor_forest.RandomForestGraphs(self.hparams)
        self.train_op = forest_graph.training_graph(self.X, self.y)
        self.loss_op = forest_graph.training_loss(self.X, self.y)
        self.infer_op = forest_graph.inference_graph(self.X)
        correct_pred = tf.equal(tf.argmax(self.infer_op, 1), tf.cast(self.y, tf.int64))
        self.acc_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # end method


    def fit(self, X, y, batch_size=1024, n_epoch=2):
        self.sess.run(tf.global_variables_initializer())
        for epoch in range(n_epoch):
            for local_step, (X_batch, y_batch) in enumerate(zip(self.next_batch(X, batch_size),
                                                                self.next_batch(y, batch_size))):
                _, loss, acc = self.sess.run([self.train_op, self.loss_op, self.acc_op],
                                             {self.X:X_batch, self.y:y_batch})
                print ("[%d / %d] [%d / %d] train_loss: %.4f | train_acc: %.4f" % 
                        (epoch+1, n_epoch, local_step, len(X)//batch_size, loss, acc))
    # end method


    def predict(self, X_test, batch_size=128):
        batch_pred_list = []
        for X_test_batch in self.next_batch(X_test, batch_size):
            batch_pred = self.sess.run(self.infer_op, {self.X:X_test_batch})
            batch_pred_list.append(batch_pred)
        return np.argmax(np.vstack(batch_pred_list), 1)
    # end method


    def next_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method
# end class
