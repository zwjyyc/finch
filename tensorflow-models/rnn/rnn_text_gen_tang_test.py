from rnn_text_gen import RNNTextGen


stopwords = ['，', '。']
prime_texts = ['长']


if __name__ == '__main__':
    with open('./temp/poetryFromTang.txt', encoding='utf-8') as f:
        text = f.read()
    
    model = RNNTextGen(text, seq_len=20, stopwords=stopwords)
    log = model.fit_text(prime_texts)