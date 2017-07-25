import string
from word2vec_skipgram import SkipGram


if __name__ == '__main__':
    with open('temp/ptb_train.txt') as f:
        text = f.read()
    sample_words = ['six', 'gold', 'japan', 'college']

    model = SkipGram(text, sample_words, useless_words=string.punctuation)
    model.fit()
