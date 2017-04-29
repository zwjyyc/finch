import re
import string
import collections
import numpy as np


def clean_text(text):
    punctuation = string.punctuation
    punctuation = ''.join([x for x in punctuation if x not in ['-', "'"]])
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


def convert_text_to_idx(all_word_list, word2idx):
    all_word_idx = []
    for word in all_word_list:
        try:
            all_word_idx.append(word2idx[word])
        except:
            all_word_idx.append(0)
    return np.array(all_word_idx)
# end function convert_text_to_idx()
