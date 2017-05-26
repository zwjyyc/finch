from rnn_text_gen import RNNTextGen


useless_words = ['，', '。']
prime_texts = ['长']


if __name__ == '__main__':
    with open('./temp/tang.txt', encoding='utf-8') as f:
        text = f.read()
    model = RNNTextGen(text, useless_words=useless_words)
    log = model.fit_text(prime_texts, text_iter_steps=50)
