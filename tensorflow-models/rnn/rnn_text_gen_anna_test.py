from rnn_text_gen import RNNTextGen


prime_texts = ['i']


if __name__ == '__main__':
    with open('./temp/anna.txt') as f:
        text = f.read()
    model = RNNTextGen(text)
    log = model.fit_text(prime_texts, text_iter_step=25)
