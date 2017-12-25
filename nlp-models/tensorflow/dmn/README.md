* Many functions are adapted from [Alex Barron's](https://github.com/barronalex/Dynamic-Memory-Networks-in-TensorFlow) work, thanks for him!
* Based on that:
    * We have added a sequence decoder in the answer module, so it can talk
```python train.py```
```
[5/5] | [150/156] | loss:0.133 | acc:0.922
final testing accuracy: 0.919

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
