from rnn_text_gen import RNNTextGen
import string
import sys


def parse(text, target_words):
    if int(sys.version[0]) >= 3:
        table = str.maketrans({w: '' for w in target_words})
        text = text.translate(table)
    else:
        for w in target_words:
            text = text.replace(w, '')


if __name__ == '__main__':
    with open('./temp/nietzsche.txt') as f:
        text = f.read()
    model = RNNTextGen(text)
    log = model.fit(['the'])
