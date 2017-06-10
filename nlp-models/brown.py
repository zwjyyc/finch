from nltk.corpus import brown
from collections import Counter


def get_indexed(vocab_size):
    sentences = brown.sents()
    flattened = [word.lower() for words in sentences for word in words]
    words = [word_freq[0] for word_freq in Counter(flattened).most_common(vocab_size)]
    word2idx = {word : idx for idx, word in enumerate(words)}
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
