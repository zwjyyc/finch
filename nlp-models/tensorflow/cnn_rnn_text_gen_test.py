from cnn_rnn_text_gen import ConvRNNTextGen


if __name__ == '__main__':
    with open('./temp/anna.txt') as f:
        text = f.read()

    model = ConvRNNTextGen(text)
    log = model.fit(start_word='the')
