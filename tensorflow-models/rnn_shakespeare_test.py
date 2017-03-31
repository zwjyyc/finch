import os
import requests
import re
import string
import collections
import numpy as np
from utils import to_one_hot
from rnn_lang_model import RNNLangModel


batch_size = 100
training_seq_len = 50


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
    # Check if file is downloaded.
    if not os.path.isfile(os.path.join(data_dir, data_file)):
        print('Not found, downloading Shakespeare texts from www.gutenberg.org')
        shakespeare_url = 'http://www.gutenberg.org/cache/epub/100/pg100.txt'
        # Get Shakespeare text
        response = requests.get(shakespeare_url)
        shakespeare_file = response.content
        # Decode binary into string
        s_text = shakespeare_file.decode('utf-8')
        # Drop first few descriptive paragraphs.
        s_text = s_text[7675:]
        # Remove newlines
        s_text = s_text.replace('\r\n', '')
        s_text = s_text.replace('\n', '')
        # Write to file
        with open(os.path.join(data_dir, data_file), 'w') as out_conn:
            out_conn.write(s_text)
    else:
        # If file has been saved, load from that file
        with open(os.path.join(data_dir, data_file), 'r') as file_conn:
            s_text = file_conn.read().replace('\n', '')
    return s_text


def build_vocab(characters):
    character_counts = collections.Counter(characters)
    # Create vocab --> index mapping
    chars = character_counts.keys()
    vocab_to_ix_dict = {key:(ix+1) for ix, key in enumerate(chars)}
    # Add unknown key --> 0 index
    vocab_to_ix_dict['_unknown']=0
    # Create index --> vocab mapping
    ix_to_vocab_dict = {val:key for key,val in vocab_to_ix_dict.items()}
    return(ix_to_vocab_dict, vocab_to_ix_dict)


def clean_text(s_text):
    s_text = re.sub(r'[{}]'.format(punctuation), ' ', s_text)
    s_text = re.sub('\s+', ' ', s_text ).strip().lower()
    return s_text


def convert_text_to_word_vecs(char_list, vocab2idx):
    s_text_ix = []
    for x in char_list:
        try:
            s_text_ix.append(vocab2idx[x])
        except:
            s_text_ix.append(0)
    s_text_ix = np.array(s_text_ix)
    return s_text_ix


def create_batch(s_text_ix):
    # Create batches for each epoch
    num_batches = int(len(s_text_ix)/(batch_size * training_seq_len)) + 1
    # Split up text indices into subarrays, of equal size
    batches = np.array_split(s_text_ix, num_batches)
    # Reshape each split into [batch_size, training_seq_len]
    batches = [np.resize(x, [batch_size, training_seq_len]) for x in batches]
    return batches


if __name__ == '__main__':
    s_text = load_text()

    print('Cleaning Text')
    s_text = clean_text(s_text)

    #char_list = list(s_text) # split up by characters
    word_list = s_text.split()

    print('Building Shakespeare Vocab by Characters')
    idx2vocab, vocab2idx = build_vocab(word_list)
    vocab_size = len(idx2vocab)
    print('Vocabulary Length = {}'.format(vocab_size))
    assert(len(idx2vocab) == len(vocab2idx)) # sanity Check

    s_text_idx = convert_text_to_word_vecs(word_list, vocab2idx)
    
    batch_list = create_batch(s_text_idx)
    X = batch_list
    y = [np.roll(batch, -1, axis=0) for batch in batch_list]
    model = RNNLangModel(n_hidden=128, n_layers=2, vocab_size=vocab_size, seq_len=training_seq_len)
    model.fit(X, y)
