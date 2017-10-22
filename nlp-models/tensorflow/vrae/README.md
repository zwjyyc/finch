```
python train.py --num_epoch 25
```
* The encoder and decoder are implemented in the latest ```tf.contrib.seq2seq``` interface (TF 1.3)
* Following features are in:
  * KL cost annealing (Bengio, 2016)
  * Word dropout and historyless decoding (Bengio, 2016)
  * Residual RNN connection
  * Beam Search
