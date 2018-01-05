---
Implementing the idea of ["Toward Controlled Generation of Text"](https://arxiv.org/abs/1703.00955)

---
<img src="https://github.com/zhedongzheng/finch/blob/master/assets/control-vae.png" height='300'>

```
python train_base_vae.py
```
```
Step 18737 | Train VAE | [24/25] | [750/781] | nll_loss:59.4 | kl_w:1.00 | kl_loss:12.13

Word Dropout
Data Shuffled

G: give it a must see if you read the book it is not for me <end>
------------
I: i love this film it is so good to watch

O: i can't understand what to say this movie is very funny and poorly executed poorly <end> -1

R: i really liked this movie a good movie is a good movie i have to <end>
------------
```
```
python train_discriminator.py
```
```
------------
Step 21415 | Train Discriminator | [3/25] | [300/781]
	| clf_loss:61.53 | clf_acc:0.74 | L_u: 5.10

Step 21415 | Train Encoder | [3/25] | [300/781]
	| seq_loss:72.0 | kl_w:1.00 | kl_loss:8.46

Step 21415 | Train Generator | [3/25] | [300/781]
	| seq_loss:71.2 | kl_w:1.00 | kl_loss:8.45
	| temperature:0.11 | l_attr_z:1.93 | l_attr_c:1.09
------------
I: i love this film it is so good to watch

O: i love this movie i highly recommend this movie to anyone who likes this movie <end>

R: i rented this movie because it was so bad but it would have been warned <end>
------------
I: this movie is horrible and waste my time

O: this is the worst movie i've ever seen this movie is a lot of time <end>

R: this movie is a charming and charming film and it's a charming movie with the <end>
------------
```
where:
* I is the encoder input

* G is the decoder output with prior z

* O is the decoder output with posterior z

* R is the decoder output with posterior z, when the attribute c (e.g. sentiment) is reversed
