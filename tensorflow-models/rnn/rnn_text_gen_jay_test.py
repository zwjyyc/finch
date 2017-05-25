from rnn_text_gen import RNNTextGen


prime_texts = ['我']


if __name__ == '__main__':
    with open('./temp/JayLyrics.txt', encoding='utf-8') as f:
        text = f.read()
    
    model = RNNTextGen(text, min_freq=1)
    log = model.fit_text(prime_texts, n_epoch=100)
