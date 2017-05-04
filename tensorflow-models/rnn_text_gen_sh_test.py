import numpy as np
import tensorflow as tf
from utils import clean_text, build_vocab, convert_text_to_idx, load_shakespeare_text
from rnn_text_gen import RNNTextGen


BATCH_SIZE = 128
SEQ_LEN = 50
NUM_LAYER = 3
CELL_SIZE = 128
RESOL = 'char'
text_iter_step = 25
num_gen = 200
prime_texts = ['thou art more',
               'to be or not to',
               'wherefore art thou']


if __name__ == '__main__':
    text = load_shakespeare_text()
    print('Cleaning text...')
    text = clean_text(text)
    if RESOL == 'char':
        all_word_list = list(text)
    if RESOL == 'word':
        all_word_list = text.split()

    word2idx, idx2word = build_vocab(all_word_list)
    vocab_size = len(idx2word)
    print('Vocabulary length:', vocab_size)
    assert len(idx2word) == len(word2idx), "len(idx2word) is not equal to len(word2idx)" # sanity Check

    all_word_idx = convert_text_to_idx(all_word_list, word2idx)
    X = np.array([all_word_idx[i : i+SEQ_LEN] for i in range(0, len(all_word_idx)-SEQ_LEN, text_iter_step)])
    Y = np.roll(X, -1, axis=1)
    Y[np.arange(len(X)-1), -1] = X[np.arange(1,len(X)), 0]
    print('X shape:', X.shape, 'Y shape:', Y.shape)
    
    sess = tf.Session()
    with tf.variable_scope('train_model'):
        train_model = RNNTextGen(cell_size=CELL_SIZE, n_layers=NUM_LAYER,
                                 vocab_size=vocab_size, seq_len=SEQ_LEN,
                                 word2idx=word2idx, idx2word=idx2word, sess=sess)
    with tf.variable_scope('train_model', reuse=True):
        sample_model = RNNTextGen(cell_size=CELL_SIZE, n_layers=NUM_LAYER,
                                  vocab_size=vocab_size, seq_len=1,
                                  word2idx=word2idx, idx2word=idx2word, sess=sess)
    log = train_model.fit(X, Y, n_epoch=25, batch_size=BATCH_SIZE,
                          en_exp_decay=True, en_shuffle=False, keep_prob=1.0,
                          sample_model=sample_model, prime_texts=prime_texts, num_gen=num_gen)
    