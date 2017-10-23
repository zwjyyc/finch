```
python train.py --rnn_cell gru
```
* The encoder and decoder are implemented in the latest ```tf.contrib.seq2seq``` interface (TF 1.3)

* Following features are enabled:

  * KL cost annealing (Bengio, 2016)
  
  * Word dropout and historyless decoding (Bengio, 2016)
  
  * Residual RNN connection
  
  * Beam Search
