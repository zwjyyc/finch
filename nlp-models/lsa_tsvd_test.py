import string
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD


lemmatizer = WordNetLemmatizer()
token2idx = {}
idx2token = {}
documents = []
token_idx = 0


def tokenize(string):
    string = string.lower()
    tokens = word_tokenize(string) # more powerful split()
    tokens = [token for token in tokens if len(token)>2] # remove too short words
    tokens = [lemmatizer.lemmatize(token) for token in tokens] # words into base form
    tokens = [token for token in tokens if token not in stopwords] # remove stopwords
    tokens = [token for token in tokens if not any(c.isdigit() for c in token)] # remove any token that contains number
    return tokens


def tokens2vec(tokens):
    vec = np.zeros(len(token2idx))
    for token in tokens:
        idx = token2idx[token]
        vec[idx] = 1
    return vec


if __name__ == '__main__':
    # rstrip() removes '\n' at the end of each line
    lines = [line.rstrip() for line in open('temp/all_book_titles.txt')]

    stopwords = set(line.rstrip() for line in open('temp/stopwords.txt')).union({
    'introduction', 'edition', 'series', 'application',
    'approach', 'card', 'access', 'package', 'plus', 'etext',
    'brief', 'vol', 'fundamental', 'guide', 'essential', 'printed',
    'third', 'second', 'fourth'
    })

    for line in lines:
        line = line.decode('ascii', 'ignore') # this is done in py2, py3 needs to test again
        tokens = tokenize(line)
        documents.append(tokens)
        for token in tokens:
            if token not in token2idx: # in tests the existence of a key in a dict
                token2idx[token] = token_idx
                idx2token[token_idx] = token
                token_idx += 1
    assert len(idx2token) == len(token2idx), "The length of idx2token is unequal to token2idx"

    X = np.zeros((len(token2idx), len(documents))) # tokens x documents
    for i, tokens in enumerate(documents):
        X[:, i] = tokens2vec(tokens)

    model = TruncatedSVD()
    X_2d = model.fit_transform(X)
    plt.scatter(X_2d[:, 0], X_2d[:, 1])
    for i in range(len(idx2token)):
        plt.annotate(s=idx2token[i], xy=(X_2d[i, 0], X_2d[i, 1]))
    plt.show()
