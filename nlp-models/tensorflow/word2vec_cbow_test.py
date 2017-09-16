import string
from word2vec_cbow import CBOW


if __name__ == '__main__':
    with open('temp/ptb_train.txt') as f:
        text = f.read()
    sample_words = ['six', 'gold', 'japan', 'college']

    model = CBOW(text, sample_words, useless_words=string.punctuation)
    model.fit()
