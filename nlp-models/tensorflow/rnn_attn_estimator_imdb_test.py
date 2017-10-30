from __future__ import absolute_import, division, print_function
from rnn_attn_estimator import model_fn
from rnn_attn_estimator_imdb_config import args
import tensorflow as tf
import numpy as np
import argparse


def zero_pad(X, max_seq_len=args.max_len):
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
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=args.vocab_size)
    X_train, X_train_lens = zero_pad(X_train)
    X_test, X_test_lens = zero_pad(X_test)
    print("Data Processed")

    tf.logging.set_verbosity(tf.logging.INFO)
    tf_estimator = tf.estimator.Estimator(model_fn)
    val_accs = []

    for epoch in range(args.num_epochs):
        tf_estimator.train(tf.estimator.inputs.numpy_input_fn(
            x={'data':X_train, 'data_len':X_train_lens}, y=y_train, batch_size=args.batch_size,
            num_epochs=1, shuffle=True))
        
        res = tf_estimator.evaluate(tf.estimator.inputs.numpy_input_fn(
            x={'data': X_test, 'data_len':X_test_lens}, y=y_test, batch_size=args.batch_size, shuffle=False))
        val_accs.append(res['test_acc'])
        
        tf_estimator = tf.estimator.Estimator(model_fn, model_dir=tf_estimator.model_dir)
    
    for epoch, val_acc in zip(range(args.num_epochs), val_accs):
        print("Epoch %d | Validation Accuracy: %.3f" % (epoch+1, val_acc))
# end function main()


if __name__ == '__main__':
    main()