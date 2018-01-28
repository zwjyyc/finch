from config import args
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from text import Tokenizer

import os
import time
import gensim
import numpy as np
import pandas as pd
import tensorflow as tf
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class BaseDataLoader(object):
    def __init__(self):
        self.data = {
            'train': {
                'X': None, 'Y': None},
            'test': {
                'X': None, 'Y': None},
            'submit': {
                'X': None}}
        self.params = {
            'n_class': None,
            'embedding': None}
        self.vocab = {
            'word2idx': None,
            'idx2word': None}

    def next_batch(self):
        """
        1. different max seq length, avoid sentence clipping
        2. consider class-balanced batch
        """
        pass


class DataLoader(BaseDataLoader):
    def __init__(self):
        super().__init__()
        self.read_files()
        self.build_vocab()
        self.index_sequences()
        self.pad_sequences()
        self.label_sequences()
        self.split_train_test()
        self.word2vec()

    def read_files(self):
        t0 = time.time()
        self.df = {
            'train': pd.read_csv('../data/train.csv'),
            'submit': pd.read_csv('../data/test.csv')}
        print("%.2f secs ==> pd.read_csv"%(time.time()-t0))

    def build_vocab(self):
        sent_list = self.df['train']['comment_text'].tolist()
        self.tokenizer = Tokenizer(args.vocab_size)
        
        t0 = time.time()
        self.tokenizer.fit_on_texts(sent_list)
        print("%.2f secs ==> tokenizer.fit_on_texts"%(time.time()-t0))
        
        self.vocab['word2idx'] = {w: i for w, i in self.tokenizer.word_index.items() if i < args.vocab_size}
        self.vocab['idx2word'] = {i: w for w, i in self.vocab['word2idx'].items()}

        print({
            w: i for w, i in self.vocab['word2idx'].items() if (i < 100 or i > (args.vocab_size - 100))})

    def index_sequences(self):
        # Training data
        sent_list = self.df['train']['comment_text'].tolist()
        t0 = time.time()
        self.X = self.tokenizer.texts_to_sequences(sent_list,
            self.vocab['word2idx'].keys())
        print("%.2f secs ==> tokenizer.texts_to_sequences"%(time.time()-t0))

        # Evaluation data
        sent_list = self.df['submit']['comment_text'].tolist()
        t0 = time.time()
        self.data['submit']['X'] = self.tokenizer.texts_to_sequences(sent_list,
            self.vocab['word2idx'].keys())
        print("%.2f secs ==> tokenizer.texts_to_sequences"%(time.time()-t0))
    
    def pad_sequences(self):
        pad_fn = lambda x: tf.keras.preprocessing.sequence.pad_sequences(
            x, args.max_len, padding='post', truncating='post')
        
        t0 = time.time()
        self.X = pad_fn(self.X)
        print("%.2f secs ==> pad_sequences"%(time.time()-t0))

        t0 = time.time()
        self.data['submit']['X'] = pad_fn(self.data['submit']['X'])
        print("%.2f secs ==> pad_sequences"%(time.time()-t0))

    def label_sequences(self):
        classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.params['n_class'] = len(classes)
        self.Y = self.df['train'][classes].values

    def split_train_test(self):
        t0 = time.time()
        (self.data['train']['X'], self.data['test']['X'],
            self.data['train']['Y'], self.data['test']['Y']) = train_test_split(
                self.X, self.Y, test_size=args.test_size)
        print("%.2f secs ==> sklearn.model_selection.train_test_split"%(time.time()-t0))

    def word2vec(self):
        path = './embedding'
        if os.path.isfile(path):
            t0 = time.time()
            model = gensim.models.Word2Vec.load(path)
            print("%.2f secs ==> gensim.models.Word2Vec"%(time.time()-t0))
        else:
            text_train = self.df['train']['comment_text'].tolist()
            sents = self.tokenizer.texts_to_sentences(text_train,
                self.vocab['word2idx'].keys())
            print(sents[-1])

            t0 = time.time()
            model = gensim.models.Word2Vec(sents, size=args.embed_dim)
            model.save(path)
            print("%.2f secs ==> gensim.models.Word2Vec"%(time.time()-t0))

        embedding = np.zeros((args.vocab_size, args.embed_dim))
        for w, i in self.vocab['word2idx'].items():
            if i != 0:
                embedding[i, :] = model[w]
        self.params['embedding'] = embedding

    def next_train_batch(self):
        X = self.data['train']['X']
        Y = self.data['train']['Y']
        for i in range(0, len(X), args.batch_size):
            yield X[i : i+args.batch_size], Y[i : i+args.batch_size]

    def next_test_batch(self):
        X = self.data['test']['X']
        Y = self.data['test']['Y']
        for i in range(0, len(X), args.batch_size):
            yield X[i : i+args.batch_size], Y[i : i+args.batch_size]

    def next_predict_batch(self):
        X = self.data['submit']['X']
        for i in range(0, len(X), args.batch_size):
            yield X[i : i+args.batch_size]


class Test:
    def __init__(self, dl):
        self.dl = dl

    def df_test(self):
        print(self.dl.df['train'].loc[1]['comment_text'])
        print('-'*12)
        print(self.dl.df['submit'].loc[2]['comment_text'])

    def idx2word_test(self):
        idx2word = self.dl.vocab['idx2word']
        X = self.dl.data['submit']['X']
        print({i: w for i, w in idx2word.items()})
        for idx in [10, 200, 1000]:
            print(' '.join([idx2word[i] for i in X[idx]]))
            print()

    def max_idx_test(self):
        X = self.dl.data['train']['X']
        print('Max Index:', max([idx for x in X for idx in x]))

    def seq_len_test(self):
        print([len(x) for x in self.dl.data['train']['X']])
        print(self.dl.data['train']['Y'].shape)

    def next_batch_test(self):
        x, y = next(self.dl.next_train_batch())
        print(x.shape, y.shape)

    def word2vec_test(self):
        print(self.dl.params['word2vec'].shape)


def main():
    test = Test(DataLoader())
    test.idx2word_test()
    #test.max_idx_test()
    #test.seq_len_test()
    test.next_batch_test()


if __name__ == '__main__':
    main()
