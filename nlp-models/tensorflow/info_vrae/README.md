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
     
     * Highway network between the encoder and decoder (gated Z)
     
     * (Optional) Mutual information loss
       * Explicitly enforce mutual information between the latent code and the generated data as part of its loss function

* Usage
   * ``` python train.py ```
    
       sampling after 54 epoches:
       ```
       Step 21074 | [54/100] | [350/390] | nll_loss:132.1 | kl_w:1.000 | kl_loss:6.31 
        I: don't get me wrong i love most of paul <unk> movies so it was with sheer excitement i was able to attend at the rolling <unk> screening at the <unk>

        D: <unk> <unk> <unk> <unk> i <unk> most of <unk> <unk> <unk> so <unk> <unk> <unk> <unk> <unk> i <unk> <unk> <unk> <unk> <unk> <unk> rolling <unk> <unk> at <unk> <unk>

        O: let me start by saying that i am a huge fan of this movie and i have to say that this movie was a <unk> of the <unk> <unk> <end>

        G: this movie was the worst movie i have ever seen in my life when i saw it at the end of the <unk> of the <unk> of the <unk> <unk>
       ```
   where:
   * I is the encoder input

   * D is the decoder input (after word dropout, most are unknown)

   * O is the decoder output with regards to encoder input

   * G is the text generation, after replacing the latent vector (z) by gaussian distribution
       * the latent vector is directly sampled from gaussian, disconnected from encoder
