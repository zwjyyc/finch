from __future__ import absolute_import, division, print_function
from rnn_attn_estimator import Estimator
import tensorflow as tf
import numpy as np
import argparse


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


def main(args):
    (X_train, y_train), (X_test, y_test) = \
        tf.contrib.keras.datasets.imdb.load_data(num_words=args.vocab_size)
    X_train, X_train_lens = zero_pad(X_train)
    X_test, X_test_lens = zero_pad(X_test)
    print("Data Processed")

    model = Estimator(args.vocab_size, args.num_classes, args.embedding_dims, args.rnn_size,
        args.dropout_rate, args.clip_norm)
    model.fit(X_train, X_train_lens, y_train, X_test, X_test_lens, y_test,
        batch_size=args.batch_size, n_epoch=args.num_epoch)
    y_pred = model.predict(X_test, X_test_lens, batch_size=args.batch_size)
    print("final testing accuracy: %.4f" % (y_pred == y_test).mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epoch', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--embedding_dims', type=int, default=100)
    parser.add_argument('--rnn_size', type=int, default=100)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--clip_norm', type=float, default=5.0)
    args = parser.parse_args()
    print(args)
    main(args)