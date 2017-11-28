* Please be warned that the VAE in NLP is much more difficult to make it work than in Computer Vision
* Features

   * The encoder and decoder are implemented in the latest ```tf.contrib.seq2seq``` interface (TF 1.3)

   * Following tricks are enabled:

     * KL cost annealing ([Bengio, 2015](https://arxiv.org/abs/1511.06349))

     * Word dropout and historyless decoding ([Bengio, 2015](https://arxiv.org/abs/1511.06349))
       * ```word_dropout_rate``` means how many percentage of decoder input words are masked with unknown tags, to encourage decoder relying on encoded information

     * To enable concatenating latent vector (z) with every input of decoder, we need to modify the decoder in original ```tf.contrib.seq2seq```;
       * The modified decoders are placed in the folder ``` modified_tf_classes ```

     * Residual RNN connection

     * Beam search
     
     * (Optional) Mutual information loss
       * Explicitly enforce mutual information between the latent code and the generated data as part of its loss function

* Usage
   * ``` python train.py ```
    
       sampling after 50 epochs:
       ```
       I: i am a big fan of stephen king i loved the running man so obviously

       D: <unk> <unk> a <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk>

       O: i am a huge fan of <unk> and i am a fan of this movie

       G: when i <unk> a <unk> to <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk>
       ```
   where:
   * I is the encoder input

   * D is the decoder input (if high word dropout is set, then most are unknown)

   * O is the decoder output with regards to encoder input

   * G is the text generation, after replacing the latent vector (z) by gaussian distribution
       * the latent vector is directly sampled from gaussian, disconnected from encoder
