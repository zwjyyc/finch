from pointer_net import PointerNetwork
import sys
import numpy as np
if int(sys.version[0]) == 2:
    from io import open


def read_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
# end function


def build_map(data):
    specials = ['<GO>',  '<EOS>', '<PAD>', '<UNK>']
    chars = list(set([char for line in data.split('\n') for char in line]))
    chars = sorted(chars)
    idx2char = {idx: char for idx, char in enumerate(chars+specials)}
    char2idx = {char: idx for idx, char in idx2char.items()}
    return idx2char, char2idx
# end function


def preprocess_data(max_len):
    X_data = read_data('temp/letters_source.txt')
    Y_data = read_data('temp/letters_target.txt')

    X_idx2char, X_char2idx = build_map(X_data)

    x_unk = X_char2idx['<UNK>']
    x_eos = X_char2idx['<EOS>']
    x_pad = X_char2idx['<PAD>']

    X_indices = []
    X_seq_len = []
    Y_indices = []
    Y_seq_len = []

    for line in X_data.split('\n'):
        chars = [X_char2idx.get(char, x_unk) for char in line]
        _chars = chars + [x_eos] + [x_pad]* (max_len-1-len(chars))
        X_indices.append(_chars)
        X_seq_len.append(len(chars)+1)
        Y_indices.append(np.argsort(_chars).tolist())
        Y_seq_len.append(len(chars)+1)

    return X_indices, X_seq_len, Y_indices, Y_seq_len, X_char2idx, X_idx2char
# end function


def train_test_split(X_indices, X_seq_len, Y_indices, Y_seq_len, BATCH_SIZE):
    X_train = X_indices[BATCH_SIZE:]
    X_train_len = X_seq_len[BATCH_SIZE:]
    Y_train = Y_indices[BATCH_SIZE:]
    Y_train_len = Y_seq_len[BATCH_SIZE:]

    X_test = X_indices[:BATCH_SIZE]
    X_test_len = X_seq_len[:BATCH_SIZE]
    Y_test = Y_indices[:BATCH_SIZE]
    Y_test_len = Y_seq_len[:BATCH_SIZE]

    return (X_train, X_train_len, Y_train, Y_train_len), (X_test, X_test_len, Y_test, Y_test_len)
# end function


def main():
    BATCH_SIZE = 128
    MAX_LEN = 15
    X_indices, X_seq_len, Y_indices, Y_seq_len, X_char2idx, X_idx2char = preprocess_data(MAX_LEN)
    
    (X_train, X_train_len, Y_train, Y_train_len), (X_test, X_test_len, Y_test, Y_test_len) \
        = train_test_split(X_indices, X_seq_len, Y_indices, Y_seq_len, BATCH_SIZE)
    
    model = PointerNetwork(
        max_len = MAX_LEN,
        rnn_size = 50,
        n_layers = 2,
        X_word2idx = X_char2idx,
        embedding_dim = 50)
    
    model.fit(X_train, X_train_len, Y_train, Y_train_len,
        val_data=(X_test, X_test_len, Y_test, Y_test_len), batch_size=BATCH_SIZE, n_epoch=50)
    model.infer('common', X_idx2char)
    model.infer('apple', X_idx2char)
    model.infer('zhedong', X_idx2char)
# end main


if __name__ == '__main__':
    main()
