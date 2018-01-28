from config import args
from sklearn.feature_extraction.text import TfidfTransformer
from text import Tokenizer

import time
import numpy as np
import pandas as pd
import tensorflow as tf


class BaseDataLoader(object):
    def __init__(self):
        self.data = {
            'train': {
                'X': None, 'Y': None},
            'submit': {
                'X': None}}
        self.params = {
            'class_list': None}

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
        self.tfidf()
        self.label_sequences()

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

        print({
            w: i for w, i in self.tokenizer.word_index.items() if ((i < 50) or ((args.vocab_size - 50) < i < args.vocab_size))})

    def index_sequences(self):
        # Training data
        sent_list = self.df['train']['comment_text'].tolist()
        t0 = time.time()
        self.data['train']['X'] = self.tokenizer.texts_to_sequences(sent_list)
        print("%.2f secs ==> tokenizer.texts_to_sequences"%(time.time()-t0))

        # Evaluation data
        sent_list = self.df['submit']['comment_text'].tolist()
        t0 = time.time()
        self.data['submit']['X'] = self.tokenizer.texts_to_sequences(sent_list)
        print("%.2f secs ==> tokenizer.texts_to_sequences"%(time.time()-t0))
    
    def tfidf(self):
        tfidf = TfidfTransformer()

        t0 = time.time()
        DT_train = self.count_matrix(self.data['train']['X'])
        self.data['train']['X'] = tfidf.fit_transform(DT_train)
        print("%.2f secs ==> TfidfTransformer().fit_transform()"%(time.time()-t0))

        DT_test = self.count_matrix(self.data['submit']['X'])
        self.data['submit']['X'] = tfidf.fit_transform(DT_test)
        print("%.2f secs ==> TfidfTransformer().fit_transform()"%(time.time()-t0))

    def count_matrix(self, X):
        t0 = time.time()
        DT = np.zeros((len(X), args.vocab_size))
        for i, indices in enumerate(X):
            for idx in indices:
                DT[i, idx] += 1
        print("%.2f secs ==> Count Matrix"%(time.time()-t0))
        return DT

    def label_sequences(self):
        self.params['class_list'] = [
            'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.data['train']['Y'] = self.df['train'][self.params['class_list']].values


def main():
    dl = DataLoader()
    print(dl.data['train']['X'].shape)
    print(dl.data['submit']['X'].shape)


if __name__ == '__main__':
    main()
