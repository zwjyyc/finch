```
python train.py --rnn_cell gru
```
* The encoder and decoder are implemented in the latest ```tf.contrib.seq2seq``` interface (TF 1.3)

* Following features are enabled:

  * KL cost annealing ([Bengio, 2015](https://arxiv.org/abs/1511.06349))
  
  * Word dropout and historyless decoding ([Bengio, 2015](https://arxiv.org/abs/1511.06349))
  
  * Residual RNN connection
  
  * Beam Search

* Sample after 20 epoches:
> I: steve smith has finally run a fairly weak series right into the ground with this movie poor actors <unk> a horrible script pretty much sums this one up two hours of your life you'll never get back go get a root <unk> instead you'll enjoy it more <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>

> O: this is one of the worst movies i have seen in a long time i have to say that this is one of the worst movies i have ever seen in my entire life i have to say that this is one of the worst movies i have ever seen in my life i have to say that this is one of the worst movies i have ever seen br br the only thing that bothered me about this <end>

> G: if you are looking for a better movie than this movie this is one of the best movies of all time it is one of the best movies i have ever seen in my opinion it is the best movie i've seen in a long time i have to say that this is one of the best movies of all time <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>

where:
* I is the encoder input

* O is the decoder output

* G is the text generation after replacing the latent vector by gaussian noise (encoder disabled)
