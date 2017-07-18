import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD


class LSA:
    def __init__(self, stopwords):
        self.stopwords = stopwords
        self.lemmatizer = WordNetLemmatizer()
        self.token2idx = {}
        self.idx2token = {}
        self.documents = []
        self.token_idx = 0
        self.X = None


    def fit(self, documents):
        for line in documents:
            line = line.decode('ascii', 'ignore') # this is done in py2, py3 needs to test again
            tokens = self.tokenize(line)
            self.documents.append(tokens)
            for token in tokens:
                if token not in self.token2idx: # tests the existence of a key in a dict
                    self.token2idx[token] = self.token_idx
                    self.idx2token[self.token_idx] = token
                    self.token_idx += 1
        assert len(self.idx2token) == len(self.token2idx), "The length of idx2token is unequal to token2idx"

        self.X = np.zeros((len(self.token2idx), len(self.documents))) # tokens x documents
        for i, tokens in enumerate(self.documents):
            self.X[:, i] = self.tokens2vec(tokens)

    
    def plot(self):
        model = TruncatedSVD()
        X_2d = model.fit_transform(self.X)
        plt.scatter(X_2d[:, 0], X_2d[:, 1])
        for i in range(len(self.idx2token)):
            plt.annotate(s=self.idx2token[i], xy=(X_2d[i, 0], X_2d[i, 1]))
        plt.show()


    def tokenize(self, string):
        string = string.lower()
        tokens = word_tokenize(string) # more powerful split()
        tokens = [token for token in tokens if len(token)>2] # remove too short words
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens] # words into base form
        tokens = [token for token in tokens if token not in self.stopwords] # remove stopwords
        tokens = [token for token in tokens if not any(c.isdigit() for c in token)] # remove any token that contains number
        return tokens


    def tokens2vec(self, tokens):
        vec = np.zeros(len(self.token2idx))
        for token in tokens:
            idx = self.token2idx[token]
            vec[idx] = 1
        return vec
