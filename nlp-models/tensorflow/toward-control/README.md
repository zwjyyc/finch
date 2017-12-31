---
Implementing the idea of Hu, et. al., ICML 2017's "Toward Controlled Generation of Text"

---
<img src="https://github.com/zhedongzheng/finch/blob/master/assets/control-vae.png" height='300'>

``` python train.py ```
```
Step 19901 | Train Generator | [6/30] | [350/781]
	| seq_loss:74.6 | kl_w:1.00 | kl_loss:7.17
	| temperature:0.01 | l_attr_z:1.93 | l_attr_c:0.49
------------
I: i love this film it is so good to watch

O: i enjoyed this movie and it was so much to watch it is a refreshing <end> -1

R: i found this movie as much else to watch this movie i was very disappointed <end>
------------
I: this movie is horrible and waste my time

O: this movie is one of the worst movies i have ever seen in this movie was

R: this movie is a great movie with a few moments in the end of the <end>
```

``` python test.py ```
```
Loading trained model ...
I: i love this film it is one of the best

O: i am a fan of this movie is one of the best of the best <end>

R: i saw this movie because i was a fan of this movie was so bad <end>
------------
I: this film is awful and the acting is bad

O: this is one of the worst movie i have ever seen this movie is not <end>

R: this is one of the best films ever made it is a refreshing to see <end>
------------
I: i hate this boring movie and there is no point to watch

O: i have seen this movie the worst film ever made in the <unk> i have <end>

R: i loved this movie because i was a chance to see this movie is refreshing <end>
------------
```
where:
* I is the encoder input

* O is the decoder output

* R is the decoder output when the attribute c (e.g. sentiment) is reversed
