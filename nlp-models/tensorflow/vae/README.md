<img src="https://github.com/zhedongzheng/finch/blob/master/assets/vrae_motivation.png" height='300'>

---
* TensorFlow >= 1.4
* scikit-learn
```
git clone https://github.com/zhedongzheng/vae-nlp.git
cd vae-nlp
```
---
```O``` is expected to write under the structure of ```I```, but with different words

``` python train.py ```
```
Step 78169 | [100/100] | [750/781] | nll_loss:66.7 | kl_w:1.000 | kl_loss:13.47 


I: this is probably the best movie from director hector it shows a brazilian reality unknown by foreigners which is the

D: <start> this <unk> probably the <unk> movie from <unk> <unk> <unk> <unk> <unk> brazilian <unk> unknown <unk> <unk> which is <unk>

O: is <unk> and the <unk> is the <unk> to end up in the <unk> that the writer director has all <end>


I: i love this film and i think it is one of the best films

O: if there was a babe and see this is the only thing that is the most of the greatest <end>


I: this movie is a waste of time and there is no point to watch it

O: and it is a <unk> to the <unk> to the <unk> for it to the multiplex to see this movie <end>

Model saved in file: ./saved/vrae.ckpt
```
where:
* I is the encoder input

* D is the decoder input (if 90% word dropout is set, then about 9 out of 10 words are unknown)

* O is the decoder output with regards to encoder input I

* G is random text generation, replacing the latent vector (z) by unit gaussian
---
Following tricks are enabled:
* KL cost annealing ([Bengio, 2015](https://arxiv.org/abs/1511.06349))

* Word dropout and historyless decoding ([Bengio, 2015](https://arxiv.org/abs/1511.06349))
    * ```word_dropout_rate``` is the % of decoder input words masked with unknown tags, in order to weaken the decoder and force it relying on encoder

* Concatenating latent vector (z) into decoder inputs, which requires modifying the decoder in source code ```tf.contrib.seq2seq```;
    * The modified decoders are placed in the folder ``` modified_tf_classes ```
---
Reference
* [Generating Sentences from a Continuous Space](https://arxiv.org/abs/1511.06349)

<img src="https://github.com/zhedongzheng/finch/blob/master/assets/vrae_struct.jpg" height='300'>

<img src="https://github.com/zhedongzheng/finch/blob/master/assets/vrae.png" height="300">
