<img src="https://github.com/zhedongzheng/finch/blob/master/assets/vrae_struct.jpg" height='300'>

<img src="https://github.com/zhedongzheng/finch/blob/master/assets/vrae.png" height="300">

* The encoder and decoder are implemented in the latest ```tf.contrib.seq2seq``` interface

* Following tricks are enabled:

    * KL cost annealing ([Bengio, 2015](https://arxiv.org/abs/1511.06349))

    * Word dropout and historyless decoding ([Bengio, 2015](https://arxiv.org/abs/1511.06349))
        * ```word_dropout_rate``` means the percentage of decoder input words are masked with unknown tags, in order to weaken the decoder and force it relying on encoded information

    * Concatenating latent vector (z) into decoder inputs, which requires modifying the decoder in source code ```tf.contrib.seq2seq```;
        * The modified decoders are placed in the folder ``` modified_tf_classes ```

* ``` python train.py ```

    it can be observed that O is trying to keep the sentence structure of I
  
    ```
    Step 19510 | [50/100] | [350/390] | nll_loss:59.4 | kl_w:1.000 | kl_loss:11.43 

    I: <unk> <unk> has never done a film so far as i know and this includes

    D: <unk> <unk> <unk> never <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk>

    O: <unk> <unk> has been a great movie as a <unk> but it was <unk> to

    G: as many years ago i have been <unk> to see this movie for this one
    ```
  where:
    * I is the encoder input

    * D is the decoder input (if 90% word dropout is set, then about 9 out of 10 words are unknown)

    * O is the decoder output with regards to encoder input

    * G is random text generation, replacing the latent vector (z) by unit gaussian
