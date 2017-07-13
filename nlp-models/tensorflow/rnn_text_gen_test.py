from rnn_text_gen import RNNTextGen


if __name__ == '__main__':
    with open('./temp/nietzsche.txt') as f:
        text = f.read()
    
    model = RNNTextGen(text)
    log = model.fit(start_word = 'the')
