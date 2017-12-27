<img src="https://github.com/zhedongzheng/finch/blob/master/assets/vrae_motivation.png" height='300'>

---
* TensorFlow >= 1.4
* scikit-learn
```
git clone https://github.com/zhedongzheng/vae-nlp.git
cd vae-nlp
```
---
Following tricks are enabled:
* KL cost annealing

* Word dropout

  ```word_dropout_rate``` is the % of decoder input words masked with unknown tags, in order to weaken the decoder and force it relying on encoder

* Concatenating latent vector (z) into decoder inputs, which requires modifying the decoder in source code ```tf.contrib.seq2seq```

  The modified decoders are placed in the folder ``` modified_tf_classes ```
---
``` python train.py ```
```
Step 23429 | [30/30] | [750/781] | nll_loss:50.5 | kl_w:1.000 | kl_loss:14.03 

G: this is the worst movie ever made the film i have ever seen in the <end>
------------
I: the 60´s is a well balanced mini series between historical facts and a good plot

D: <start> <unk> <unk> <unk> <unk> <unk> balanced <unk> <unk> <unk> historical <unk> <unk> a <unk> plot

O: the movie is one of the most interesting and the actors and a great cast <end>
------------
I: i love this film and i think it is one of the best films

O: i love this movie and i found it was a fan of the best films <end>
------------
I: this movie is a waste of time and there is no point to watch it

O: this movie is a complete waste of time to this movie and don't miss it <end>
------------
Model saved in file: ./saved/vrae.ckpt
```
where:
* I is the encoder input

* D is the decoder input (if 90% word dropout is set, then about 9 out of 10 words are unknown)

* O is the decoder output with regards to encoder input I

* G is random text generation, replacing the latent vector (z) by unit gaussian

---
Reference
* [Generating Sentences from a Continuous Space](https://arxiv.org/abs/1511.06349)

<img src="https://github.com/zhedongzheng/finch/blob/master/assets/vrae_struct.jpg" height='300'>

<img src="https://github.com/zhedongzheng/finch/blob/master/assets/vrae.png" height="300">
