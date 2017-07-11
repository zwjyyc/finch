from cnn_rnn_text_gen import ConvRNNTextGen
import string


prime_texts = ['the']


if __name__ == '__main__':
    with open('./temp/anna.txt') as f:
        text = f.read()

    useless_words = list(string.punctuation.replace(',', '').replace('.',''))

    model = ConvRNNTextGen(text,
                           stopwords = [',', '.'],
                           useless_words = useless_words,
    )
    log = model.fit(prime_texts)
