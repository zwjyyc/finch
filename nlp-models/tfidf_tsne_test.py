from brown import get_indexed
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfTransformer
from utils import find_analogies
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    indexed, word2idx = get_indexed(min_freq=50)
    vocab_size = len(word2idx)
    print("Data loaded | Vocab size:", vocab_size, '| Document size:', len(indexed))

    X = np.zeros((vocab_size, len(indexed))) # term-document matrix
    j = 0
    for document in indexed:
        for idx in document:
            X[idx, j] += 1
        j += 1
    print("Term-Document matrix built ...")

    model = TfidfTransformer()
    X = model.fit_transform(X)
    X = X.toarray()
    print("TF-IDF transform completed ...")

    idx2word = {idx : word for word, idx in word2idx.iteritems()}

    model = TSNE(n_components=3, verbose=2)
    X = model.fit_transform(X)
    print("TSNE transform completed ...")

    find_analogies('king', 'man', 'woman', X, word2idx)

    plt.scatter(X[:,0], X[:,1])
    for i in range(vocab_size):
        try:
            plt.annotate(s=idx2word[i], xy=(X[i,0], X[i,1]))
        except:
            print("bad string:", idx2word[i])
    plt.show()
