from __future__ import print_function
from brown import get_indexed
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np


def find_closest(input_words, word_embedding, word2idx, idx2word):
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
# end function find_closest


if __name__ == '__main__':
    indexed, word2idx = get_indexed(10000)
    vocab_size = len(word2idx)
    print("Data loaded | Vocab size:", vocab_size, '| Document size:', len(indexed))

    DT = np.zeros((len(indexed), vocab_size)) # document-term matrix
    for i, indices in enumerate(indexed):
        for idx in indices:
            DT[i, idx] += 1
    print("Document-Term matrix built ...")

    model = TfidfTransformer()
    DT = model.fit_transform(DT).toarray()
    TD = DT.T
    print("TF-IDF transform completed ...")

    idx2word = {idx : word for word, idx in word2idx.items()}
    find_closest(['london', 'king', 'italy', 'queen'], TD, word2idx, idx2word)
