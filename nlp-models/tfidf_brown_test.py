from brown import get_indexed
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfTransformer
from utils import find_analogy
import numpy as np
import matplotlib.pyplot as plt


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

    find_analogy('london', TD, word2idx)
    find_analogy('king', TD, word2idx)
    find_analogy('italy', TD, word2idx)
    find_analogy('queen', TD, word2idx)
