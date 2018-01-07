---
Implementing the idea of ["Toward Controlled Generation of Text"](https://arxiv.org/abs/1703.00955)

---
<img src="https://github.com/zhedongzheng/finch/blob/master/assets/control-vae.png" height='300'>

```
python train_base_vae.py
```
```
Step 18737 | Train VAE | [24/25] | [750/781] | nll_loss:48.7 | kl_w:1.00 | kl_loss:15.90

Word Dropout
Data Shuffled

G: they are all the other films they were trying to be afraid of a film <end>
------------
I: i love this film it is so good to watch

O: to say that this movie is the worst movie i would give it 0 10 <end>

R: i really enjoyed this movie and i didn't want to write a copy of it <end>
------------
```
```
python train_discriminator.py
```
```
------------
Step 21115 | Train Discriminator | [3/25] | [0/781]
	| clf_loss:45.84 | clf_acc:0.85 | L_u: 5.21

Step 21115 | Train Encoder | [3/25] | [0/781]
	| seq_loss:63.1 | kl_w:1.00 | kl_loss:11.35

Step 21115 | Train Generator | [3/25] | [0/781]
	| seq_loss:64.6 | kl_w:1.00 | kl_loss:11.22
	| temperature:0.12 | l_attr_z:0.44 | l_attr_c:1.23
------------
I: i love this film it is so good to watch

O: there are so many years to see this movie it was very good for me <end>

R: i think this movie was the worst movie i've ever seen br br the plot <end>
------------
I: this movie is horrible and waste my time

O: this movie was awful awful it was nothing better than this movie was the worst <end>

R: this film is amazing and it was very good and see it for the masterpiece <end>
------------
```
where:
* I is the encoder input

* G is the decoder output with prior z

* O is the decoder output with posterior z

* R is the decoder output with posterior z, when the attribute c (e.g. sentiment) is reversed
