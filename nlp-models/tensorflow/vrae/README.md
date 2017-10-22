```
python train.py
```
* The encoder and decoder are implemented in the latest ```tf.contrib.seq2seq``` interface
* Following features are in:
  * KL cost annealing (Bengio, 2016)
  * Word dropout and historyless decoding (Bengio, 2016)
  * Residual RNN connection
  * Beam Search
