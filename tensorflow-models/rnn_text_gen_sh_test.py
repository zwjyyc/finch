import numpy as np
import tensorflow as tf
from utils import clean_text, build_vocab, convert_text_to_idx, load_shakespeare_text
from rnn_text_gen import RNNTextGen


BATCH_SIZE = 128
SEQ_LEN = 50
NUM_LAYER = 3
CELL_SIZE = 128
RESOL = 'char'
prime_texts = [
    'thou art more',
    'to be or not to',
    'wherefore art thou'
]


if __name__ == '__main__':
    text = load_shakespeare_text()
    print('Cleaning Text')
    text = clean_text(text)
    if RESOL == 'char':
        all_word_list = list(text)
    if RESOL == 'word':
        all_word_list = text.split()

    print('Building Shakespeare Vocab by Characters')
    idx2word, word2idx = build_vocab(all_word_list)
    vocab_size = len(idx2word)
    print('Vocabulary Length = {}'.format(vocab_size))
    assert len(idx2word) == len(word2idx), "len(idx2word) is not equal to len(word2idx)" # sanity Check

    all_word_idx = convert_text_to_idx(all_word_list, word2idx)
    X = np.resize(all_word_idx, [int(len(all_word_idx)/SEQ_LEN), SEQ_LEN])
    print('X shape: ', X.shape)
    
    sess = tf.Session()
    with tf.variable_scope('train_model'):
        train_model = RNNTextGen(cell_size=CELL_SIZE, n_layers=NUM_LAYER,
                                 vocab_size=vocab_size, seq_len=SEQ_LEN, resolution=RESOL,
                                 sess=sess)
    with tf.variable_scope('train_model', reuse=True):
        sample_model = RNNTextGen(cell_size=CELL_SIZE, n_layers=NUM_LAYER, resolution=RESOL,
                                  vocab_size=vocab_size, seq_len=1,
                                  sess=sess)
    log = train_model.fit(X, n_epoch=25, batch_size=BATCH_SIZE,
                          en_exp_decay=True, en_shuffle=False,
                          sample_pack=(sample_model, idx2word, word2idx, 20, prime_texts))
    