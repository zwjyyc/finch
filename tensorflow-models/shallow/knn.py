import tensorflow as tf
import numpy as np
from collections import Counter


class KNN:
    def __init__(self, n_features, k=5, sess=tf.Session(),
                 dist_fn=lambda tr,te: tf.reduce_sum(tf.abs(tf.subtract(tr, tf.expand_dims(te,1))), 2)):
        self.n_features = n_features
        self.k = k
        self.dist_fn = dist_fn
        self.sess = sess
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.X_train = tf.placeholder(tf.float32, shape=[None, self.n_features])
        self.X_test = tf.placeholder(tf.float32, shape=[None, self.n_features])
        self.y_train = tf.placeholder(tf.int32, shape=[None])

        self.dists = self.dist_fn(self.X_train, self.X_test)
        _, self.top_k_idx = tf.nn.top_k(tf.negative(self.dists), self.k)
        self.top_k_y = tf.gather(self.y_train, self.top_k_idx)
    # end method


    def predict(self, X_train, y_train, X_test, batch_size=1024):
        preds = []
        self.sess.run(tf.global_variables_initializer())
        next_batches = zip(self.next_batch(X_train, batch_size), self.next_batch(y_train, batch_size),
                           self.next_batch(X_test, batch_size))
        for i, (X_train_batch, y_train_batch, X_test_batch) in enumerate(next_batches):
            top_k_ys = self.sess.run(self.top_k_y, {self.X_train: X_train_batch, self.y_train: y_train_batch,
                                                    self.X_test: X_test_batch})
            for top_k_y in top_k_ys:
                preds.append(Counter(top_k_y).most_common()[0][0])
            print("[%d / %d]" % (i+1, len(X_test)//batch_size+1))
        return np.array(preds)
    # end method


    def next_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method
# end class