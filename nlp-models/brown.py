from nltk.corpus import brown
from collections import Counter


def get_indexed(min_freq=50):
    sentences = brown.sents()
    flattened = [word.lower() for words in sentences for word in words]
    word2freq = {word : freq for word, freq in Counter(flattened).items() if freq > min_freq}
    word2idx = {word : idx for idx, word in enumerate(word2freq.keys())}
    indexed = []
    for words in sentences:
        indexed_words = []
        for word in words:
            try:
                indexed_words.append(word2idx[word.lower()])
            except:
                continue
        indexed.append(indexed_words)
    return indexed, word2idx


if __name__ == "__main__":
    indexed, word2idx = get_indexed()
    print(indexed)
