from rnn_vae import RNN_VAE
import sys
if int(sys.version[0]) == 2:
    from io import open


def read_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
# end function read_data


def build_map(data):
    specials = ['<GO>',  '<EOS>', '<PAD>', '<UNK>']
    chars = list(set([char for line in data.split('\n') for char in line]))
    idx2char = {idx: char for idx, char in enumerate(specials + chars)}
    char2idx = {char: idx for idx, char in idx2char.items()}
    return idx2char, char2idx
# end function build_map


def preprocess_data():
    X_data = read_data('temp/letters_target.txt')
    X_idx2char, X_char2idx = build_map(X_data)
    X_indices = [[X_char2idx.get(char, X_char2idx['<UNK>']) for char in line] for line in X_data.split('\n')]
    return X_indices, X_char2idx, X_idx2char
# end function preprocess_data


def main():
    BATCH_SIZE = 128
    X_indices, X_char2idx, X_idx2char = preprocess_data()
    X_train = X_indices[BATCH_SIZE:]
    X_test = X_indices[:BATCH_SIZE]

    model = RNN_VAE(rnn_size=50, n_layers=1, X_word2idx=X_char2idx, embedding_dim=15)
    model.fit(X_train, X_test, batch_size=BATCH_SIZE)
    model.infer('abcdefg', X_idx2char)
    model.infer('ahisuvz', X_idx2char)
    model.infer('bbeehmm', X_idx2char)
    model.infer('rsuawxy', X_idx2char)
# end function main


if __name__ == '__main__':
    main()
