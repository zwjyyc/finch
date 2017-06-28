import re
import sys
import math
import numpy as np
import tensorflow as tf
from collections import Counter


class SkipGram:
    def __init__(self, text, sample_words, skip_window=5, embedding_dim=200, n_sampled=100, min_freq=5,
                 useless_words=None, loss_fn=tf.nn.sampled_softmax_loss, sess=tf.Session()):
        self.text = text
        self.sample_words = sample_words
        self.skip_window = skip_window
        self.embedding_dim = embedding_dim
        self.n_sampled = n_sampled
        self.min_freq = min_freq
        self.useless_words = useless_words
        self.loss_fn = loss_fn
        self.sess = sess
        self.preprocess_text()
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.add_input_layer()       
        self.add_word_embedding()
        self.add_backward_path()
        self.add_sampler()
    # end method build_graph


    def add_input_layer(self):
        self.x = tf.placeholder(tf.int32, shape=[None])
        self.y = tf.placeholder(tf.int32, shape=[None, 1])
        self.w = tf.get_variable('softmax_w', [self.vocab_size, self.embedding_dim], tf.float32,
                                  tf.truncated_normal_initializer(stddev=0.1))
        self.b = tf.get_variable('softmax_b', [self.vocab_size], tf.float32, tf.constant_initializer(0.01))
    # end method add_input_layer


    def add_word_embedding(self):
        self.embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_dim], -1.0, 1.0))
        self.embedded = tf.nn.embedding_lookup(self.embedding, self.x) # forward activation of the input network
    # end method add_word_embedding


    def add_backward_path(self):
        self.loss = tf.reduce_mean(self.loss_fn(weights=self.w, biases=self.b,
                                                labels=self.y, inputs=self.embedded,
                                                num_sampled=self.n_sampled, num_classes=self.vocab_size))
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
    # end method add_backward_path


    def add_sampler(self):
        self.sample_indices = [self.word2idx[w] for w in self.sample_words]
        sample_indices = tf.constant(self.sample_indices, dtype=tf.int32)

        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embedding), 1, keep_dims=True))
        normalized_embedding = self.embedding / norm # divided by modulus for cosine similarity

        sample_embedded = tf.nn.embedding_lookup(normalized_embedding, sample_indices)
        self.similarity = tf.matmul(sample_embedded, tf.transpose(normalized_embedding))
    # end method add_sampler


    def preprocess_text(self):
        text = self.text
        text = text.replace('\n', ' ')
        if self.useless_words is not None:
            if sys.version[0] == 3:
                table = str.maketrans({useless: ' ' for useless in self.useless_words})
                text = text.translate(table)
            else:
                text = re.sub(r'[{}]'.format(''.join(self.useless_words)), ' ', text)
        text = re.sub('\s+', ' ', text).strip().lower()
        
        words = text.split()
        word2freq = Counter(words)
        words = [word for word in words if word2freq[word] > self.min_freq]
        print("Total words:", len(words))

        _words = set(words)
        self.word2idx = {c: i for i, c in enumerate(_words)}
        self.idx2word = {i: c for i, c in enumerate(_words)}
        self.vocab_size = len(self.idx2word)
        print('Vocabulary size:', self.vocab_size)

        indexed = [self.word2idx[w] for w in words]
        self.indexed = self.filter_high_freq(indexed)
        print("Word preprocessing completed ...")
    # end method preprocess_text


    def filter_high_freq(self, int_words):
        t = 1e-5
        threshold = 0.8

        int_word_counts = Counter(int_words)
        total_count = len(int_words)
        
        word_freqs = {w: float(c) / total_count for w, c in int_word_counts.items()}
        prob_drop = {w: 1 - np.sqrt(t / word_freqs[w]) for w in int_word_counts}
        train_words = [w for w in int_words if prob_drop[w] < threshold]

        return train_words
    # end method filter_high_freq


    def next_batch(self, batch_size):
        for idx in range(0, len(self.indexed), batch_size):
            x, y = [], []
            batch = self.indexed[idx: idx+batch_size]
            for i in range(len(batch)):
                batch_x = batch[i]
                batch_y = self.get_y(batch, i)
                x.extend([batch_x] * len(batch_y))
                y.extend(batch_y)
            yield (x, y)
    # end method next_batch


    def get_y(self, words, idx):
        skip_window = np.random.randint(1, self.skip_window+1)
        left = idx - skip_window if (idx - skip_window) > 0 else 0
        right = idx + skip_window
        y = words[left: idx] + words[idx+1: right+1]
        return list(set(y))
    # end method get_y


    def fit(self, n_epoch=10, batch_size=1000, top_k=5, eval_step=1000):
        self.sess.run(tf.global_variables_initializer())
        global_step = 0
        n_batch = int(len(self.indexed) / batch_size)
        total_steps = int(n_epoch * n_batch)

        for epoch in range(n_epoch):
            for local_step, (x, y) in enumerate(self.next_batch(batch_size)):
                y = np.array(y)[:, np.newaxis]
                _, loss = self.sess.run([self.train_op, self.loss], {self.x: x, self.y: y})
                if local_step % 50 == 0:
                    print ('Epoch %d/%d | Batch %d/%d | train loss: %.4f' %
                           (epoch+1, n_epoch, local_step, n_batch, loss))
                global_step += 1
                if local_step % eval_step == 0:
                    similarity = self.sess.run(self.similarity)
                    for i in range(len(self.sample_words)):
                        analogies = (-similarity[i]).argsort()[1:top_k+1]
                        log = 'Nearest to [%s]:' % self.idx2word[self.sample_indices[i]]
                        for k in range(top_k):
                            analogy = self.idx2word[analogies[k]]
                            log = '%s %s,' % (log, analogy)
                        print(log)
    # end method fit
# end class