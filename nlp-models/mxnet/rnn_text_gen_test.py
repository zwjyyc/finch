from rnn_text_gen import RNNTextGen
import mxnet as mx


if __name__ == '__main__':
    with open('./temp/anna.txt') as f:
        text = f.read()
    
    model = RNNTextGen(mx.cpu(), text)
    log = model.fit(start_word = 'The ')
