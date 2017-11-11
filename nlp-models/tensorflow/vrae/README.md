```
python train.py --rnn_cell gru
```
* The encoder and decoder are implemented in the latest ```tf.contrib.seq2seq``` interface (TF 1.3)

* Following features are enabled:

  * KL cost annealing ([Bengio, 2015](https://arxiv.org/abs/1511.06349))
  
  * Word dropout and historyless decoding ([Bengio, 2015](https://arxiv.org/abs/1511.06349))
    * By default, we set ```word_dropout_rate``` to 1.0, which means no inputs are presented to decoder to maximise dependency on encoder

  * To enable concatenating latent vector (z) with every input of decoder, we need to modify the decoder in original ```tf.contrib.seq2seq```;
    * The modified decoders are placed in the folder ``` modified_tf_classes ```
  
  * Residual RNN connection
  
  * Beam Search

* Sample after 20 epoches:
  > I: i would not deny that i have quite enjoyed watching any japanese horror films but everyone must get quite fed up with them after you have seen the same thing

  > D: <unk> would not deny <unk> i have quite enjoyed watching <unk> japanese horror <unk> but <unk> <unk> get <unk> fed up with them after you have seen <unk> <unk> thing

  > O: i have to admit that i have to say that this show was one of those films that i have seen in a long time when it came out <end>

  > G: this movie is about a group of teens who are sent to the <unk> of the <unk> of the <unk> they are sent to the <unk> of the <unk> <end>

where:
* I is the encoder input

* D is the decoder input (after word dropout, in default setting they are all unknown)

* O is the decoder output with regards to encoder input

* G is the text generation, after replacing the latent vector (z) by random normal noise
    * the encoder is unused, the text is directly generated from gaussian space
