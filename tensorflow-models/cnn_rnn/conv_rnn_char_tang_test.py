from conv_rnn_char import ConvLSTMChar

stopwords = ['，', '。']

if __name__ == '__main__':
    with open('./temp/poetryFromTang.txt', encoding='utf-8') as f:
        text = f.read()
    
    model = ConvLSTMChar(text, seq_len=20, stopwords=stopwords)
    log = model.fit_text()