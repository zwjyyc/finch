class Seq2Seq:
    def __init__(self, rnn_size, n_layers,
                 X_vocab_size, encoder_embedding_dim,
                 Y_word2idx, decoder_embedding_dim,
                 batch_size,
                 ):
        self.rnn_size = rnn_size
        self.n_layers = n_layers

        self.X_vocab_size = X_vocab_size
        self.encoder_embedding_dim = encoder_embedding_dim

        self.Y_word2idx = Y_word2idx
        self.decoder_embedding_dim = decoder_embedding_dim

        self.batch_size = batch_size

        self.build_graph()
    # end constructor


    def build_graph(self):
        self.add_input_layer()
        self.add_encoder_layer()
        self.add_decoder_layer()
    # end method build_graph


    def add_input_layer(self):
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, None])
        self.X_seq_len = tf.placeholder(tf.int32, [None])
        self.Y_seq_len = tf.placeholder(tf.int32, [None])
    # end method add_input_layer


    def add_encoder_layer(self):
        def lstm_cell():
            return tf.nn.rnn_cell.LSTMCell(self.rnn_size, initializer=tf.orthogonal_initializer)
        _, self.encoder_state = tf.nn.dynamic_rnn(
            tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(self.n_layers)]), 
            tf.contrib.layers.embed_sequence(self.X, self.X_vocab_size, self.encoder_embedding_dim),
            sequence_length = self.X_seq_len,
            dtype = tf.float32,
        )
    # end method add_encoder_layer
    

    def add_decoder_layer(self):
        def process_decoder_input(data, word2idx, batch_size):
            ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1]) # remove last char
            decoder_input = tf.concat([tf.fill([batch_size, 1], word2idx['<GO>']), ending], 1)
            return decoder_input
        decoder_input = process_decoder_input(self.Y, self.Y_word2idx, self.batch_size)

        Y_vocab_size = len(self.Y_word2idx)
        decoder_embedding = tf.get_variable('decoder_embedding', tf.float32,
                                            [Y_vocab_size, self.decoder_embedding_dim],
                                            tf.random_uniform_initializer(-1.0, 1.0))
        decoder_embed_input = tf.nn.embedding_lookup(decoder_embedding, decoder_input)
    # end method add_decoder_layer
