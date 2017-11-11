```
python train.py --rnn_cell gru
```
* The encoder and decoder are implemented in the latest ```tf.contrib.seq2seq``` interface (TF 1.3)

* Following features are enabled:

  * KL cost annealing ([Bengio, 2015](https://arxiv.org/abs/1511.06349))
  
  * Word dropout and historyless decoding ([Bengio, 2015](https://arxiv.org/abs/1511.06349))
    * By default, we set ```word_dropout_rate``` to 1.0, which means no inputs are presented to decoder to maximise its dependency on encoder

  * To enable concatenating latent vector (z) with every input of decoder, we need to modify the decoder in original ```tf.contrib.seq2seq```;
    * The modified decoders are placed in the folder ``` modified_tf_classes ```
  
  * Residual RNN connection
  
  * Beam Search

* Sample after 20 epoches:
```
I: i sat through this turkey because i hadn't seen it

D: <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk> <unk>

O: i sat seeing this movie when i i it <end>

G: i <unk> a <unk> <unk> a a <unk> of <end>
```
where:
* I is the encoder input

* D is the decoder input (after word dropout, in default setting they are all unknown)

* O is the decoder output with regards to encoder input

* G is the text generation, after replacing the latent vector (z) by random normal noise
    * the text is directly generated from latent layer, disconnected from encoder
