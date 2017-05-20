from rnn_text_gen import RNNTextGen


prime_texts = ['你要离开我知道很']


if __name__ == '__main__':
    with open('./temp/JayLyrics.txt', encoding='utf-8') as f:
        text = f.read()
    
    model = RNNTextGen(text, n_layer=3, min_freq=1)
    log = model.fit_text(prime_texts)
