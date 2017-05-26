from rnn_text_gen import RNNTextGen
from string import punctuation, digits, ascii_letters


useless_words = ['，', '。', '（', '）', '☆', '-'] + list(punctuation) + list(digits) + list(ascii_letters)
prime_texts = ['我']


if __name__ == '__main__':
    with open('./temp/jaychou.txt', encoding='utf-8') as f:
        text = f.read()
    model = RNNTextGen(text, useless_words=useless_words)
    log = model.fit_text(prime_texts)
