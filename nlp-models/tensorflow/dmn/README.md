* Many functions are adapted from [Alex Barron's](https://github.com/barronalex/Dynamic-Memory-Networks-in-TensorFlow) work, thanks for him!
* Based on that:

    * We have added a decoder in the answer module for talking
    * We have reproduced ```AttentionGRUCell``` from official ```GRUCell``` from TF 1.4

```
python train.py
```
```
[10/10] | [150/156] | loss:0.009 | acc:1.000
final testing accuracy: 0.995

[['Fred', 'picked', 'up', 'the', 'football', 'there', '<end>'],
 ['Fred', 'gave', 'the', 'football', 'to', 'Jeff', '<end>'],
 ['Bill', 'went', 'back', 'to', 'the', 'bathroom', '<end>'],
 ['Jeff', 'grabbed', 'the', 'milk', 'there', '<end>'],
 ['Jeff', 'gave', 'the', 'football', 'to', 'Fred', '<end>'],
 ['Fred', 'handed', 'the', 'football', 'to', 'Jeff', '<end>'],
 ['Jeff', 'handed', 'the', 'football', 'to', 'Fred', '<end>'],
 ['Fred', 'gave', 'the', 'football', 'to', 'Jeff', '<end>']]

['Who', 'did', 'Fred', 'give', 'the', 'football', 'to', '?']

['Jeff', '<end>']

- - - - - - - - - - - - 
['Jeff', '<end>']
```
