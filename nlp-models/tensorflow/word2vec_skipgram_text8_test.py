import string
from word2vec_skipgram import SkipGram


if __name__ == '__main__':
    with open('temp/text8.txt') as f:
        text = f.read()
    sample_words = ['one', 'game', 'american', 'college']

    model = SkipGram(text, sample_words, useless_words=string.punctuation)
    model.fit()
