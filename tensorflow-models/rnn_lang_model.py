import tensorflow as tf
import math
import numpy as np


class RNNLangModel:
    def __init__(self, n_hidden, n_layers, vocab_size, seq_len, sess):
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.sess = sess
        self.build_graph()
    # end constructor


    def build_graph(self):
        with tf.variable_scope('input_layer'):
            self.add_input_layer()            
        with tf.variable_scope('forward_path'):
            self.add_word_embedding_layer()
            self.add_lstm_cells()
            self.add_dynamic_rnn()
            self.reshape_rnn_out()
        with tf.name_scope('output_layer'):
            self.add_output_layer()
        with tf.name_scope('backward_path'):
            self.add_backward_path()
    # end method build_graph


    def add_input_layer(self):
        self.batch_size = tf.placeholder(tf.int32)
        self.X = tf.placeholder(tf.int32, [None, self.seq_len])
        self.Y = tf.placeholder(tf.int32, [None, self.seq_len])
        self.W = tf.get_variable('W', [self.n_hidden, self.vocab_size], tf.float32,
                                 tf.contrib.layers.variance_scaling_initializer())
        self.b = tf.get_variable('b', [self.vocab_size], tf.float32, tf.constant_initializer(0.0))
    # end method add_input_layer


    def add_word_embedding_layer(self):
        """
        X from (batch_size, seq_len) -> (batch_size, seq_len, n_hidden)
        where each word in (batch_size, seq_len) is represented by a vector of length [n_hidden]
        """
        embedding_mat = tf.get_variable('embedding_mat', [self.vocab_size, self.n_hidden], tf.float32,
                                        tf.random_normal_initializer())
        self.embedding_out = tf.nn.embedding_lookup(embedding_mat, self.X)
    # end method add_word_embedding_layer


    def add_lstm_cells(self):
        cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)
        self.cells = tf.contrib.rnn.MultiRNNCell([cell] * self.n_layers)
    # end method add_rnn_cells


    def add_dynamic_rnn(self):
        self.init_state = self.cells.zero_state(self.batch_size, tf.float32)
        self.rnn_out, self.final_state = tf.nn.dynamic_rnn(self.cells, self.embedding_out,
                                                           initial_state=self.init_state, time_major=False)       
    # end method add_dynamic_rnn


    def reshape_rnn_out(self):
        self.rnn_out = tf.reshape(self.rnn_out, [-1, self.n_hidden])
    # end method add_rnn_out


    def add_output_layer(self):
        self.logits = tf.nn.bias_add(tf.matmul(self.rnn_out, self.W), self.b)
        self.softmax_out = tf.nn.softmax(self.logits)
    # end method add_output_layer


    def add_backward_path(self):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                labels=tf.reshape(self.Y, [-1]))
        self.loss = tf.reduce_mean(losses)
        self.lr = tf.placeholder(tf.float32)
        """
        gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tf.trainable_variables()), 4.5)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))
        """
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    # end method add_backward_path


    def adjust_lr(self, en_exp_decay, global_step, n_epoch, nb_batch):
        if en_exp_decay:
            max_lr = 0.003
            min_lr = 0.0001
            decay_rate = math.log(min_lr/max_lr) / (-n_epoch*nb_batch)
            lr = max_lr*math.exp(-decay_rate*global_step)
        else:
            lr = 0.001
        return lr
    # end method adjust_lr


    def fit(self, X_batch_list, Y_batch_list, n_epoch=10, batch_size=128, en_exp_decay=True, sample_pack=None):
        log = {'train_loss': []}
        global_step = 0
        self.sess.run(tf.global_variables_initializer()) # initialize all variables
        if sample_pack is not None:
            s_model, idx2word, word2idx, num, prime_texts = sample_pack
        for epoch in range(n_epoch):
            next_state = self.sess.run(self.init_state, feed_dict={self.batch_size:batch_size})
            batch_count = 1
            for X_batch, Y_batch in zip(X_batch_list, Y_batch_list):
                lr = self.adjust_lr(en_exp_decay, global_step, n_epoch, len(X_batch_list))
                _, loss, next_state = self.sess.run([self.train_op, self.loss, self.final_state],
                    feed_dict={self.X:X_batch, self.Y:Y_batch, self.init_state:next_state,
                               self.batch_size:batch_size, self.lr:lr})
                if batch_count % 10 == 0:
                    print ('Epoch %d/%d | Batch %d/%d | train loss: %.4f | lr: %.4f' % (epoch+1, n_epoch,
                        batch_count, len(X_batch_list), loss, lr))
                log['train_loss'].append(loss)
                batch_count += 1
                global_step += 1
            if sample_pack is not None:
                for prime_text in prime_texts:
                    print(self.sample(s_model, idx2word, word2idx, num, prime_text))
        return log
    # end method fit


    def sample(self, s_model, idx2word, word2idx, num, prime_text):
        state = self.sess.run(s_model.init_state, feed_dict={s_model.batch_size:1})
        word_list = prime_text.split()
        for word in word_list[:-1]:
            x = np.zeros((1,1))
            x[0,0] = word2idx[word] 
            state = self.sess.run(s_model.final_state, feed_dict={s_model.X:x, s_model.init_state:state})

        out_sentence = prime_text
        word = word_list[-1]
        for n in range(num):
            x = np.zeros((1,1))
            x[0,0] = word2idx[word]
            softmax_out, state = self.sess.run([s_model.softmax_out, s_model.final_state],
                                                feed_dict={s_model.X:x, s_model.init_state:state})
            sample = np.argmax(softmax_out[0])
            if sample == 0:
                break
            word = idx2word[sample]
            out_sentence = out_sentence + ' ' + word
        return(out_sentence)
    # end method sample
# end class CharRNNModel
