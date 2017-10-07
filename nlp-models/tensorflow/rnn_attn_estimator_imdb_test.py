from __future__ import absolute_import, division, print_function
from rnn_attn_estimator import Estimator
import tensorflow as tf
import numpy as np


def zero_pad(X, max_seq_len=250):
    sequences = []
    sequence_lens = []
    for x in X:
        if len(x) >= max_seq_len:
            sequences.append(x[:max_seq_len])
            sequence_lens.append(max_seq_len)
        if len(x) < max_seq_len:
            sequences.append(x + [0]*(max_seq_len-len(x)))
            sequence_lens.append(len(x))
    return np.array(sequences), np.array(sequence_lens)


def main():
    VOCAB_SIZE = 20000
    BATCH_SIZE = 32
    N_EPOCH = 2

    (X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)
    X_train, X_train_lens = zero_pad(X_train)
    X_test, X_test_lens = zero_pad(X_test)
    print("Data Processed")

    model = Estimator(VOCAB_SIZE, 2)
    model.fit(X_train, X_train_lens, y_train, batch_size=BATCH_SIZE, n_epoch=N_EPOCH)
    model.score(X_test, X_test_lens, y_test, batch_size=BATCH_SIZE)


if __name__ == '__main__':
    main()