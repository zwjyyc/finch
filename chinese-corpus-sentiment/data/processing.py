#!/usr/bin/env python
# -*- coding: utf-8 -*-
import fasttext
import os
import jieba
import time
import cPickle as pickle


class ChiWordMapper:
    def __init__(self):
        self.data_base_path = os.path.join(os.path.dirname(os.getcwd()), 'data')
        self.corpus_base_path = os.path.join(self.data_base_path, 'ChnSentiCorp_htl_ba_6000')
        t = time.time()
        self.fasttext_model = self.load_fasttext_bin( '/home/zhedongzheng/Dataset/zh.bin' )
        print "Model loaded: %.2f secs" % (time.time() - t)
        self.removable_list = ['，', '．', '\n', '。', '！', '、']
        self.words_limit = 20


    def load_data(self):
        docums_path_dict = self.get_docums_path_dict()
        X, y = self.gen_X_y_tensors(docums_path_dict)
        X = self.clip_docum_len(X)
        return X, y
        

    def load_fasttext_bin(self, path):
        return fasttext.load_model(path)


    def get_docums_path_dict(self):
        pos_docums_path = sorted( os.listdir( os.path.join(self.corpus_base_path, 'pos') ) )
        neg_docums_path = sorted( os.listdir( os.path.join(self.corpus_base_path, 'neg') ) )
        return {'pos': pos_docums_path, 'neg': neg_docums_path}

    
    def gen_X_y_tensors(self, docums_path_dict):
        X = []
        for polarity in ('pos', 'neg'):
            for idx, docum_path in enumerate( docums_path_dict[polarity] ):
                vectors_for_each_docum = []
                with open( os.path.join( self.corpus_base_path, polarity, docum_path ) ) as f:
                    document = f.read()
                    document = self.remove_stopwords(document)
                    seg_list = list(jieba.cut(document, cut_all=False))
                    for seg in seg_list:
                        vectors_for_each_docum.append(self.fasttext_model[seg])
                X.append(vectors_for_each_docum)
                print "Document %s %d %s" % (polarity, idx, docum_path)
        y = [1] * len(docums_path_dict['pos']) + [0] * len(docums_path_dict['neg'])
        print "X: %d | y: %d" % (len(X), len(y))
        return X, y
    

    def remove_stopwords(self, string):
        for rem in self.removable_list:
            string = string.replace(rem, '')
        return string


    def clip_docum_len(self, tensor): # to control the width of each row to be the same
        for idx, document in enumerate(tensor):
            if len(document) != self.words_limit:
                if len(document) > self.words_limit:
                    tensor[idx] = tensor[idx][:self.words_limit]
                    print "Docum %d" % idx
                else:
                    for _ in xrange(self.words_limit - len(document)):
                        tensor[idx].append([0.0] * 300)
                    print "Docum %d" % idx
        return tensor


    def save_data(self):
        docums_path_dict = self.get_docums_path_dict()
        X, y = self.gen_X_y_tensors(docums_path_dict)
        X = self.clip_docum_len(X)
        with open( os.path.join(self.data_base_path, 'Xy.pkl'), 'wb' ) as output:
            pickle.dump((X, y), output, pickle.HIGHEST_PROTOCOL)
        print "Saved Successfully !"
        return os.path.join(self.data_base_path, 'Xy.pkl')

