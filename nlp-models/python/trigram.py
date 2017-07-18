import numpy as np
from nltk.tokenize import word_tokenize


class Trigram:
    def __init__(self, ):
        self.trigram = {}
        self.trigram2proba = {}
    # end constructor


    def fit(self, documents):
        for sentence in documents:
            tokens = word_tokenize(sentence.lower())
            for i in range(len(tokens) - 2):
                key = (tokens[i], tokens[i+2])
                if key not in self.trigram:
                    self.trigram[key] = []
                self.trigram[key].append(tokens[i+1])

        for key, words in self.trigram.iteritems():
            if len(set(words)) > 1:
                word2proba = {}
                for word in words:
                    if word not in word2proba:
                        word2proba[word] = 1
                    word2proba[word] += 1
                total_count = sum(list(word2proba.values()))
                for word, count in word2proba.iteritems():
                    word2proba[word] = float(count) / total_count
                self.trigram2proba[key] = word2proba
    # end method fit


    def predict(self, key):
        token2proba = self.trigram2proba[key]
        probas = list(token2proba.values())
        tokens = token2proba.keys()
        idx = np.argmax(np.random.multinomial(1, probas, size=1)[0])
        return tokens[idx]
    # end method predict
