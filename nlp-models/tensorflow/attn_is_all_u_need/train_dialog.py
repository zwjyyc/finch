from model import tf_estimator_model_fn
from config import args
from data import DataLoader

import json
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def main():
    dl = DataLoader(
        source_path='temp/dialog_source.txt',
        target_path='temp/dialog_target.txt')
    sources, targets = dl.load()
    print('Source Vocab Size:', len(dl.source_word2idx))
    print('Target Vocab Size:', len(dl.target_word2idx))
    
    tf_estimator = tf.estimator.Estimator(
        tf_estimator_model_fn, params=_prepare_params(dl), model_dir=args.model_dir)
    
    for epoch in range(args.num_epochs):
        tf_estimator.train(tf.estimator.inputs.numpy_input_fn(
            x = {'source':sources, 'target':targets},
            batch_size = args.batch_size,
            num_epochs = None,
            shuffle = True), steps=2000)
        stupid_decode(['你是谁', '你喜欢我吗', '给我唱一首歌', '我帅吗'], tf_estimator, dl)


def stupid_decode(test_words, tf_estimator, dl):
    test_indices = []
    for test_word in test_words:
        test_idx = [dl.source_word2idx[c] for c in test_word] + \
                   [dl.source_word2idx['<pad>']] * (args.source_max_len - len(test_word))
        test_indices.append(test_idx)
    test_indices = np.atleast_2d(test_indices)
    
    pred_ids = np.zeros([len(test_words), args.target_max_len], np.int64)
    for j in range(args.target_max_len):
        _pred_ids = tf_estimator.predict(tf.estimator.inputs.numpy_input_fn(
            x={'source':test_indices, 'target':pred_ids}, batch_size=len(test_words), shuffle=False))
        _pred_ids = np.array(list(_pred_ids))
        pred_ids[:, j] = _pred_ids[:, j]
    
    target_idx2word = {i: w for w, i in dl.target_word2idx.items()}
    for i, test_word in enumerate(test_words):
        ans = ''.join([target_idx2word[id] for id in pred_ids[i]])
        print(test_word, '->', ans)


def _prepare_params(dl):
    params = {
        'source_vocab_size': len(dl.source_word2idx),
        'target_vocab_size': len(dl.target_word2idx),
        'start_symbol': dl.target_word2idx['<start>'],
        'activation': _get_activation()}
    return params


def _get_activation():
    if args.activation == 'relu':
        activation = tf.nn.relu
    elif args.activation == 'elu':
        activation = tf.nn.elu
    elif args.activation == 'lrelu':
        activation = tf.nn.leaky_relu
    else:
        raise ValueError("acitivation fn has to be 'relu' or 'elu' or 'lrelu'")
    return activation


if __name__ == '__main__':
    print(json.dumps(args.__dict__, indent=4))
    main()