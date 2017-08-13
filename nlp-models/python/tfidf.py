from __future__ import print_function
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np


class Tfidf:
    def __init__(self):
        self.model = TfidfTransformer()

    
    def fit(self, documents_indexed, vocab_size):
        DT = np.zeros((len(documents_indexed), vocab_size)) # document-term matrix
        for i, indices in enumerate(documents_indexed):
            for idx in indices:
                DT[i, idx] += 1
        print("Document-Term matrix built ...")

        model = TfidfTransformer()
        DT = model.fit_transform(DT).toarray()
        TD = DT.T
        print("TF-IDF transform completed ...")
        return TD

    
    def find_closest(self, input_words, word_embedding, word2idx, idx2word):
        """
        def euclidean_dist(a, b):
            return np.linalg.norm(a - b)
        """
        def similarity(vec, embedding):
            a = vec
            b = embedding.T
            return - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b, axis=0))

        input_vecs = [word_embedding[word2idx[input_word]] for input_word in input_words]
        for input_vec, input_word in zip(input_vecs, input_words):
            score = similarity(input_vec, word_embedding)
            best_word = idx2word[score.argsort()[1]]
            print("closest match by: ", input_word, ' - ', best_word)
