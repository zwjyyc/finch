from conv_rnn_char import ConvLSTMChar


if __name__ == '__main__':
    with open('./temp/JayLyrics.txt', encoding='utf-8') as f:
        text = f.read()
    
    model = ConvLSTMChar(text, min_freq=1)
    log = model.fit_text()
