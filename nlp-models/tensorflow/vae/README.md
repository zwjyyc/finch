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
* KL cost annealing ([Bengio, 2015](https://arxiv.org/abs/1511.06349))

* Word dropout and historyless decoding ([Bengio, 2015](https://arxiv.org/abs/1511.06349))

  ```word_dropout_rate``` is the % of decoder input words masked with unknown tags, in order to weaken the decoder and force it relying on encoder

* Concatenating latent vector (z) into decoder inputs, which requires modifying the decoder in source code ```tf.contrib.seq2seq```

  The modified decoders are placed in the folder ``` modified_tf_classes ```
---
``` python train.py ```
```
Step 23429 | [30/100] | [750/781] | nll_loss:76.2 | kl_w:1.000 | kl_loss:12.23 


I: it's not exactly progressive but it's funny and inoffensive and definitely a step up from the previous year's the birdcage

D: <start> it's <unk> <unk> progressive <unk> it's funny <unk> inoffensive <unk> definitely a step <unk> <unk> <unk> <unk> <unk> the <unk>

O: for it to prevent him to the viewer to see the film with the lives of the 21st century <end>


I: i love this film and i think it is one of the best films

O: about it to be desired but it's a must see for all in all the rest of your life <end>


I: this movie is a waste of time and there is no point to watch it

O: for the end for the end of the story of this film is worth checking out if you are <end>

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
