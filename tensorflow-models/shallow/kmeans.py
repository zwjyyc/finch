import tensorflow as tf
import numpy as np


class KMeans:
    def __init__(self, k, n_features, n_classes, sess=tf.Session()):
        self.k = k
        self.n_features = n_features
        self.n_classes = n_classes
        self.sess = sess
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.n_features])
        self.Y = tf.placeholder(tf.float32, shape=[None, self.n_classes])
        params = tf.contrib.factorization.KMeans(
            inputs=self.X,
            num_clusters=self.k,
            distance_metric='cosine',
            use_mini_batch=True)
        (_, cluster_idx, scores, _, self.init_op, self.train_op) = params.training_graph()
        self.cluster_idx = cluster_idx[0]
        self.avg_distance = tf.reduce_mean(scores)
        self.cluster_label = None # to be filled after calling fit()
    # end method


    def fit(self, X, Y, n_iter=50):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.init_op, {self.X: X})
        for i in range(n_iter):
            _, d, idx = self.sess.run([self.train_op, self.avg_distance, self.cluster_idx], {self.X: X})
            print("Step %i, Avg Distance: %f" % (i, d))
        counts = np.zeros(shape=(self.k, self.n_classes))
        for i, k in enumerate(idx):
            counts[k] += Y[i]
        labels_map = [np.argmax(c) for c in counts] # assign the most frequent label to the centroid
        self.cluster_label = tf.nn.embedding_lookup(tf.convert_to_tensor(labels_map), self.cluster_idx)
    # end method


    def predict(self, X_test):
        return self.sess.run(self.cluster_label, {self.X:X_test})
    # end method
# end class