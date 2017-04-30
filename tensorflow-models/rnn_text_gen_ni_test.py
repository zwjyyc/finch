import numpy as np
import tensorflow as tf
from keras.utils.data_utils import get_file
from utils import clean_text, build_vocab, convert_text_to_idx
from rnn_text_gen import RNNTextGen


BATCH_SIZE = 32
SEQ_LEN = 30
NUM_LAYER = 3
CELL_SIZE = 128
RESOL = 'char'
prime_texts = [
    'having kept a sharp',
    'it seems to me',
    'it has gradually'
]


if __name__ == '__main__':
    path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
    text = open(path).read().replace('\n', ' ')

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
    X = np.resize(all_word_idx, [int(len(all_word_idx)/SEQ_LEN), SEQ_LEN])
    print('X shape:', X.shape)
    
    sess = tf.Session()
    with tf.variable_scope('train_model'):
        train_model = RNNTextGen(cell_size=CELL_SIZE, n_layers=NUM_LAYER,
                                 vocab_size=vocab_size, seq_len=SEQ_LEN, resolution=RESOL,
                                 word2idx=word2idx, idx2word=idx2word, sess=sess)
    with tf.variable_scope('train_model', reuse=True):
        sample_model = RNNTextGen(cell_size=CELL_SIZE, n_layers=NUM_LAYER,
                                  vocab_size=vocab_size, seq_len=1, resolution=RESOL,
                                  word2idx=word2idx, idx2word=idx2word, sess=sess)
    log = train_model.fit(X, n_epoch=25, batch_size=BATCH_SIZE,
                          en_exp_decay=True, en_shuffle=False,
                          sample_model=sample_model, prime_texts=prime_texts, num_gen=30)
    