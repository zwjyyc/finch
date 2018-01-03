---
Implementing the idea of ["Toward Controlled Generation of Text"](https://arxiv.org/abs/1703.00955)

---
<img src="https://github.com/zhedongzheng/finch/blob/master/assets/control-vae.png" height='300'>

``` python train.py ```
```
Step 15991 | Train Generator | [1/30] | [350/781]
	| seq_loss:70.4 | kl_w:1.00 | kl_loss:8.61
	| temperature:0.16 | l_attr_z:2.10 | l_attr_c:1.41
------------
I: i love this film it is so good to watch

O: i agree with this movie that i am a huge fan of the movie that <end>

R: this is the worst movie i've ever seen in a long long time i can <end>
------------
I: this movie is horrible and waste my time

O: this movie is so painful to watch this movie it is a waste of time <end>

R: this movie is a great film that it is excellent excellent br br 8 10 <end>
------------
```
where:
* I is the encoder input

* O is the decoder output

* R is the decoder output when the attribute c (e.g. sentiment) is reversed
