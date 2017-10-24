```
python train.py --rnn_cell gru
```
* The encoder and decoder are implemented in the latest ```tf.contrib.seq2seq``` interface (TF 1.3)

* Following features are enabled:

  * KL cost annealing (Bengio, 2016)
  
  * Word dropout and historyless decoding (Bengio, 2016)
  
  * Residual RNN connection
  
  * Beam Search

* Sample after 10 epoches:
```
I: i really liked tom <unk> <unk> you just have to let it come over you and enjoy it while it lasts and don't expect anything it's like sitting on a <unk> <unk> with a beer in the summer sun and watching the people go by it definitely won't keep you
```
```
O: i am a huge fan of this movie and i have to say that this is one of the worst movies i have ever seen br br the story is about a group of young people who have to get a chance to see this movie it is one of
```
```
G: one of the best movies i have ever seen br br this is a very good movie it is a very good movie about a group of young people who get a chance to get a chance to get a chance to see the <unk> of the <unk> and the
```
