# -*- coding: utf-8 -*-
"""
Vocab size: 4532
Train 29240 samples | Test 7288 samples
Epoch 1/1 | Step 0/228 | train_loss: 1.3943 | train_acc: 0.2592 | lr: 0.0050
Epoch 1/1 | Step 50/228 | train_loss: 0.4623 | train_acc: 0.8291 | lr: 0.0030
Epoch 1/1 | Step 100/228 | train_loss: 0.3521 | train_acc: 0.8770 | lr: 0.0018
Epoch 1/1 | Step 150/228 | train_loss: 0.3412 | train_acc: 0.8789 | lr: 0.0011
Epoch 1/1 | Step 200/228 | train_loss: 0.2663 | train_acc: 0.9031 | lr: 0.0007
Epoch 1/1 | train_loss: 0.3519 | train_acc: 0.8746 | test_loss: 0.3390 | test_acc: 0.8794 | lr: 0.0005
我 来到 大学 读书 ， 希望 学到 知识 
"""
import sys
import chseg
import numpy as np
import tensorflow as tf
from birnn_crf_clf import BiRNN_CRF
from collections import Counter


SEQ_LEN = 50
N_CLASS = 4 # B: 0, M: 1, E: 2, S: 3
N_EPOCH = 1
sample = '我来到大学读书，希望学到知识'
py = int(sys.version[0])


def to_seq(*args):
    data = []
    for x in args:
        x = x[: (len(x) - len(x) % SEQ_LEN)]
        data.append(np.reshape(x, [-1, SEQ_LEN]))
    return data


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, vocab_size, char2idx, idx2char = chseg.load_data()
    X_train, X_test, Y_train, Y_test = to_seq(x_train, x_test, y_train, y_test)
    print('Vocab size: %d' % vocab_size)

    clf = BiRNN_CRF(SEQ_LEN, vocab_size, N_CLASS)
    clf.fit(X_train, Y_train, val_data=(X_test, Y_test), n_epoch=N_EPOCH)
    
    chars = list(sample) if py == 3 else list(sample.decode('utf-8'))
    labels = clf.infer([char2idx[c] for c in chars])
    res = ''
    for i, l in enumerate(labels):
        c = sample[i] if py == 3 else sample.decode('utf-8')[i]
        if l == 2 or l == 3:
            c += ' '
        res += c
    print(res)
    