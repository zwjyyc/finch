import os
import requests
import re
import string
import collections
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def clean_text(text):
    text = text.replace('\n', ' ')
    punctuation = string.punctuation
    punctuation = ''.join([x for x in punctuation if x not in ['-', "'"]])
    text = re.sub(r'[{}]'.format(punctuation), ' ', text)
    text = re.sub('\s+', ' ', text ).strip().lower()
    return text
# end function clean_text()


def build_vocab(word_list, min_word_freq=None):
    word_counts = collections.Counter(word_list)
    if min_word_freq is not None:
        word_counts = {key:val for key,val in word_counts.items() if val > min_word_freq}
    words = word_counts.keys()
    word2idx = {key:(idx+1) for idx,key in enumerate(words)} # create word -> index mapping
    word2idx['_unknown'] = 0 # add unknown key -> 0 index
    idx2word = {val:key for key,val in word2idx.items()} # create index -> word mapping
    return(word2idx, idx2word)
# end function build_vocab()


def convert_text_to_idx(all_word_list, word2idx):
    all_word_idx = []
    for word in all_word_list:
        try:
            all_word_idx.append(word2idx[word])
        except:
            all_word_idx.append(0)
    return all_word_idx
# end function convert_text_to_idx()


def load_shakespeare_text():
    data_dir = 'temp'
    data_file = 'shakespeare.txt'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    print('Loading Shakespeare Data')
    if not os.path.isfile(os.path.join(data_dir, data_file)):   # check if file is downloaded
        print('Not found, downloading Shakespeare texts from www.gutenberg.org')
        shakespeare_url = 'http://www.gutenberg.org/cache/epub/100/pg100.txt'
        response = requests.get(shakespeare_url)
        shakespeare_file = response.content
        s_text = shakespeare_file.decode('utf-8')               # decode binary -> string
        s_text = s_text[7675:]                                  # drop first few descriptive paragraphs
        with open(os.path.join(data_dir, data_file), 'w') as f: # write to file
            f.write(s_text)
    else:
        with open(os.path.join(data_dir, data_file), 'r') as f: # if file has been saved, load from that file
            s_text = f.read()
    return s_text
# end function load_shakespeare_text()


def plot(log, dir='./log'):
    if not os.path.exists(dir):
        os.makedirs(dir)
    sns.set(style='white')
    plt.plot(log['loss'], label='train_loss')
    plt.plot(log['val_loss'], label='test_loss')
    plt.legend(loc='best')
    plt.savefig(os.path.join(dir, sys.argv[0][:-3]))
    print("Figure created !")
# end function plot()
