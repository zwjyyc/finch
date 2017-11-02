from model import tf_estimator_model_fn
from config import args
from data import DataLoader

import numpy as np
import tensorflow as tf


def main():
    dl = DataLoader(
        source_path='temp/letters_source.txt',
        target_path='temp/letters_target.txt')
    sources, targets = dl.load()

    params = {
        'source_vocab_size': len(dl.source_word2idx),
        'target_vocab_size': len(dl.target_word2idx),
        'start_symbol': dl.symbols.index('<start>')}
    tf_estimator = tf.estimator.Estimator(tf_estimator_model_fn, params=params)

    tf.logging.set_verbosity(tf.logging.INFO)
    tf_estimator.train(tf.estimator.inputs.numpy_input_fn(
        x={'source':sources, 'target':targets}, batch_size=args.batch_size, num_epochs=args.num_epochs,
        shuffle=True))
    
    stupid_decode_test('apple', tf_estimator, dl)
    stupid_decode_test('common', tf_estimator, dl)


def stupid_decode_test(test_word, tf_estimator, dl, test_maxlen=7):
    test_idx = [dl.source_word2idx[c] for c in test_word] + \
            [dl.source_word2idx['<pad>']] * (test_maxlen - len(test_word))
    test_idx = np.atleast_2d(test_idx)
    
    pred_ids = np.zeros([1, test_maxlen], np.int64)
    for j in range(test_maxlen):
        _pred_ids = tf_estimator.predict(tf.estimator.inputs.numpy_input_fn(
            x={'source':test_idx, 'target':pred_ids}, batch_size=1, shuffle=False))
        _pred_ids = np.array(list(_pred_ids))
        pred_ids[:, j] = _pred_ids[:, j]
    
    target_idx2word = {i: w for w, i in dl.target_word2idx.items()}
    ans = ''.join([target_idx2word[id] for id in pred_ids[0]])
    print(test_word, '->', ans)


if __name__ == '__main__':
    main()