import os, sys, requests
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from utils import clean_text, build_vocab, convert_text_to_idx
from rnn_text_gen import RNNTextGen


BATCH_SIZE = 100
SEQ_LEN = 50
NUM_LAYER = 3
CELL_SIZE = 128
prime_texts = ['thou art more', 'to be or not to', 'wherefore art thou']


def load_text():
    data_dir = 'temp'
    data_file = 'shakespeare.txt'
    model_path = 'shakespeare_model'
    full_model_dir = os.path.join(data_dir, model_path)
    if not os.path.exists(full_model_dir):
        os.makedirs(full_model_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    print('Loading Shakespeare Data')
    if not os.path.isfile(os.path.join(data_dir, data_file)): # check if file is downloaded
        print('Not found, downloading Shakespeare texts from www.gutenberg.org')
        shakespeare_url = 'http://www.gutenberg.org/cache/epub/100/pg100.txt'
        response = requests.get(shakespeare_url) # get Shakespeare text
        shakespeare_file = response.content
        s_text = shakespeare_file.decode('utf-8') # decode binary into string
        s_text = s_text[7675:] # drop first few descriptive paragraphs
        s_text = s_text.replace('\r\n', '') # remove newlines
        s_text = s_text.replace('\n', '') # remove newlines
        with open(os.path.join(data_dir, data_file), 'w') as out_conn: # write to file
            out_conn.write(s_text)
    else:
        with open(os.path.join(data_dir, data_file), 'r') as file_conn: # If file has been saved, load from that file
            s_text = file_conn.read().replace('\n', '')
    return s_text
# end function load_text()


if __name__ == '__main__':
    text = load_text()
    print('Cleaning Text')
    text = clean_text(text)
    all_word_list = list(text) # now we are doing char-level, use .split() for word-level

    print('Building Shakespeare Vocab by Characters')
    idx2word, word2idx = build_vocab(all_word_list)
    vocab_size = len(idx2word)
    print('Vocabulary Length = {}'.format(vocab_size))
    assert len(idx2word) == len(word2idx), "len(idx2word) is not equal to len(word2idx)" # sanity Check

    all_word_idx = convert_text_to_idx(all_word_list, word2idx)
    X = np.resize(all_word_idx, [int(len(all_word_idx)/SEQ_LEN), SEQ_LEN])
    
    sess = tf.Session()
    with tf.variable_scope('training'):
        train_model = RNNTextGen(cell_size=CELL_SIZE, n_layers=NUM_LAYER,
                                 vocab_size=vocab_size, seq_len=SEQ_LEN,
                                 sess=sess)
    with tf.variable_scope('training', reuse=True):
        sample_model = RNNTextGen(cell_size=CELL_SIZE, n_layers=NUM_LAYER,
                                  vocab_size=vocab_size, seq_len=1,
                                  sess=sess)
    log = train_model.fit(X, n_epoch=10, batch_size=BATCH_SIZE,
                          en_exp_decay=True,
                          sample_pack=(sample_model, idx2word, word2idx, 30, prime_texts))
    