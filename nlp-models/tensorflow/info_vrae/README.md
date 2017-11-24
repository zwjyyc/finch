* Features

   * The encoder and decoder are implemented in the latest ```tf.contrib.seq2seq``` interface (TF 1.3)

   * Following features are enabled:

     * KL cost annealing ([Bengio, 2015](https://arxiv.org/abs/1511.06349))

     * Word dropout and historyless decoding ([Bengio, 2015](https://arxiv.org/abs/1511.06349))
       * ```word_dropout_rate``` means how many percentage of decoder input words are masked with unknown tags, to encourage decoder relying on encoded information

     * To enable concatenating latent vector (z) with every input of decoder, we need to modify the decoder in original ```tf.contrib.seq2seq```;
       * The modified decoders are placed in the folder ``` modified_tf_classes ```

     * Residual RNN connection

     * Beam search
     
     * Mutual information loss
       * Explicitly enforce mutual information between the latent code and the generated data as part of its loss function
     
     * Convolutional Encoder

* Usage
   * ``` python train.py ```
   * decoding after  epoches:
       ```
        I: this has got to be the worse move i've ever

        D: <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk>

        O: this has to be be the worst film i have

        G: in the <unk> of <unk> <unk> are a lot of
       ```
   where:
   * I is the encoder input

   * D is the decoder input (after word dropout, most are unknown)

   * O is the decoder output with regards to encoder input

   * G is the text generation, after replacing the latent vector (z) by random normal noise
       * the text is directly generated from latent layer, disconnected from encoder
