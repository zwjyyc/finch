import tensorflow as tf
import math
import numpy as np
import sklearn


class RNNTextGen:
    def __init__(self, seq_len, vocab_size, cell_size, n_layers, sess):
        """
        Parameters:
        -----------
        seq_len: int
            Sequence length
        vocab_size: int
            Vocabulary size
        cell_size: int
            Number of units in the rnn cell
        n_layers: int
            Number of layers of stacked rnn cells
        sess: object
            tf.Session() object 
        """
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.cell_size = cell_size
        self.n_layers = n_layers
        self.sess = sess
        self.current_layer = None
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.add_input_layer()
                    
        self.add_word_embedding_layer()
        self.add_lstm_cells()
        self.add_dynamic_rnn()
        self.reshape_rnn_out()

        self.add_output_layer()
        self.add_backward_path()
    # end method build_graph


    def add_input_layer(self):
        self.batch_size = tf.placeholder(tf.int32)
        self.X = tf.placeholder(tf.int32, [None, self.seq_len])
        self.Y = tf.placeholder(tf.int32, [None, self.seq_len])
        self.W = tf.get_variable('W', [self.cell_size, self.vocab_size], tf.float32,
                                 tf.contrib.layers.variance_scaling_initializer())
        self.b = tf.get_variable('b', [self.vocab_size], tf.float32, tf.constant_initializer(0.0))
        self.current_layer = self.X
    # end method add_input_layer


    def add_word_embedding_layer(self):
        """
        X from (batch_size, seq_len) -> (batch_size, seq_len, n_hidden)
        where each word in (batch_size, seq_len) is represented by a vector of length [n_hidden]
        """
        embedding_mat = tf.get_variable('embedding_mat', [self.vocab_size, self.cell_size], tf.float32,
                                        tf.random_normal_initializer())
        embedding_out = tf.nn.embedding_lookup(embedding_mat, self.current_layer)
        self.current_layer = embedding_out
    # end method add_word_embedding_layer


    def add_lstm_cells(self):
        cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size)
        self.cells = tf.contrib.rnn.MultiRNNCell([cell] * self.n_layers)
    # end method add_rnn_cells


    def add_dynamic_rnn(self):
        self.init_state = self.cells.zero_state(self.batch_size, tf.float32)
        self.current_layer, self.final_state = tf.nn.dynamic_rnn(self.cells, self.current_layer,
                                                                 initial_state=self.init_state,
                                                                 time_major=False)    
    # end method add_dynamic_rnn


    def reshape_rnn_out(self):
        self.current_layer = tf.reshape(self.current_layer, [-1, self.cell_size])
    # end method add_rnn_out


    def add_output_layer(self):
        self.logits = tf.nn.bias_add(tf.matmul(self.current_layer, self.W), self.b)
    # end method add_output_layer


    def add_backward_path(self):
        losses = tf.contrib.seq2seq.sequence_loss(
            logits = tf.reshape(self.logits, [self.batch_size, self.seq_len, self.vocab_size]),
            targets = self.Y,
            weights = tf.ones([self.batch_size, self.seq_len]),
            average_across_timesteps = True,
            average_across_batch = True,
        )
        self.loss = tf.reduce_sum(losses)
        self.lr = tf.placeholder(tf.float32)
        gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tf.trainable_variables()), 4.5)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))
    # end method add_backward_path


    def decrease_lr(self, en_exp_decay, global_step, n_epoch, nb_batch):
        if en_exp_decay:
            max_lr = 0.003
            min_lr = 0.0001
            decay_rate = math.log(min_lr/max_lr) / (-n_epoch*nb_batch)
            lr = max_lr*math.exp(-decay_rate*global_step)
        else:
            lr = 0.0005
        return lr
    # end method adjust_lr


    def fit(self, X, n_epoch=10, batch_size=128, en_exp_decay=True, sample_pack=None):
        log = {'train_loss': []}
        global_step = 0
        self.sess.run(tf.global_variables_initializer()) # initialize all variables
        if sample_pack is not None:
            s_model, idx2word, word2idx, num_pred, prime_texts = sample_pack
        
        for epoch in range(n_epoch):
            next_state = self.sess.run(self.init_state, feed_dict={self.batch_size:batch_size})
            batch_count = 1
            for X_batch in self.gen_batch(X, batch_size):
                Y_batch = np.roll(X_batch, -1, axis=1)
                lr = self.decrease_lr(en_exp_decay, global_step, n_epoch, int(len(X)/batch_size))
                if len(X_batch) == batch_size:
                    _, loss, next_state = self.sess.run([self.train_op, self.loss, self.final_state],
                                                         feed_dict={self.X:X_batch, self.Y:Y_batch,
                                                                    self.init_state:next_state,
                                                                    self.batch_size:len(X_batch), self.lr:lr})
                else:
                    _, loss = self.sess.run([self.train_op, self.loss],
                                             feed_dict={self.X:X_batch, self.Y:Y_batch,
                                                        self.batch_size:len(X_batch), self.lr:lr})
                if batch_count % 10 == 0:
                    print ('Epoch %d/%d | Batch %d/%d | train loss: %.4f | lr: %.4f' % (epoch+1, n_epoch,
                    batch_count, (len(X)/batch_size), loss, lr))
                log['train_loss'].append(loss)
                batch_count += 1
                global_step += 1
            
            if sample_pack is not None:
                for prime_text in prime_texts:
                    print(self.sample(s_model, idx2word, word2idx, num_pred, prime_text), end='\n')
            
        return log
    # end method fit


    def sample(self, s_model, idx2word, word2idx, num_pred, prime_text):
        next_state = self.sess.run(s_model.init_state, feed_dict={s_model.batch_size:1})
        #word_list = prime_text.split()
        word_list = list(prime_text)
        for word in word_list[:-1]:
            x = np.zeros([1,1])
            x[0,0] = word2idx[word] 
            next_state = self.sess.run(s_model.final_state, feed_dict={s_model.X:x,
                                                                       s_model.init_state:next_state})

        out_sentence = prime_text
        word = word_list[-1]
        for n in range(num_pred):
            x = np.zeros([1,1])
            x[0,0] = word2idx[word]
            logits, next_state = self.sess.run([s_model.logits, s_model.final_state],
                                                     feed_dict={s_model.X:x, s_model.init_state:next_state})
            idx = np.argmax(logits[0])
            if idx == 0:
                break
            word = idx2word[idx]
            #out_sentence = out_sentence + ' ' + word
            out_sentence = out_sentence + word
        return(out_sentence)
    # end method sample

    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method gen_batch
# end class
