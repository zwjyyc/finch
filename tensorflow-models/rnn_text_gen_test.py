import os
import sys
import requests
import re
import string
import collections
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from rnn_text_gen import RNNTextGen


BATCH_SIZE = 100
SEQ_LEN = 50
NUM_LAYER = 3
CELL_SIZE = 128
prime_texts = ['thou art more', 'to be or not to', 'wherefore art thou']


punctuation = string.punctuation
punctuation = ''.join([x for x in punctuation if x not in ['-', "'"]])


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


def clean_text(text):
    text = re.sub(r'[{}]'.format(punctuation), ' ', text)
    text = re.sub('\s+', ' ', text ).strip().lower()
    return text
# end function clean_text()


def build_vocab(word_list, min_word_freq=5):
    word_counts = collections.Counter(word_list)
    word_counts = {key:val for key,val in word_counts.items() if val>min_word_freq}
    words = word_counts.keys()
    word2idx = {key:(idx+1) for idx,key in enumerate(words)} # create word --> index mapping
    word2idx['_unknown'] = 0 # add unknown key --> 0 index
    idx2word = {val:key for key,val in word2idx.items()} # create index --> word mapping
    return(idx2word, word2idx)
# end function build_vocab()


def convert_text_to_word_vec(all_word_list, word2idx):
    all_word_idx = []
    for word in all_word_list:
        try:
            all_word_idx.append(word2idx[word])
        except:
            all_word_idx.append(0)
    return np.array(all_word_idx)
# end function convert_text_to_word_vecs()


def plot(log, dir='./log'):
    if not os.path.exists(dir):
        os.makedirs(dir)
    sns.set(style='white')
    plt.plot(log['train_loss'])
    plt.savefig(os.path.join(dir, sys.argv[0][:-3]))
    print("Figure created !")
# end function plot()


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

    all_word_idx = convert_text_to_word_vec(all_word_list, word2idx)
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
    plot(log)
    