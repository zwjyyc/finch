# -*- coding: utf-8 -*-
import sys
import chseg
import numpy as np
import tensorflow as tf
from birnn_seq2seq_clf import BiRNN
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

    clf = BiRNN(vocab_size, N_CLASS)
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
    