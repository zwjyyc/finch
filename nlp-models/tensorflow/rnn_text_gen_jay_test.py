# -*- coding: utf-8 -*-
from rnn_text_gen import RNNTextGen
from io import open


if __name__ == '__main__':
    with open('./temp/jay.txt', encoding='utf-8') as f:
        text = f.read()
    
    model = RNNTextGen(text)
    log = model.fit(start_word = u'你要离开', n_gen=100)
