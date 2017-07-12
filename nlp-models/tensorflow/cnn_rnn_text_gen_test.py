from cnn_rnn_text_gen import ConvRNNTextGen


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

    model = ConvRNNTextGen(text)
    log = model.fit(['the'])
