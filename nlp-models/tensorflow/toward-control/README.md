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

```
where:
* I is the encoder input

* O is the decoder output

* R is the decoder output when the attribute c (e.g. sentiment) is reversed
