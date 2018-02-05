from seq2seq_attn import Seq2Seq
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
    x_data = read_data('temp/letters_source.txt')
    y_data = read_data('temp/letters_target.txt')

    x_idx2char, x_char2idx = build_map(x_data)
    y_idx2char, y_char2idx = build_map(y_data)

    x_unk = x_char2idx['<UNK>']
    y_unk = y_char2idx['<UNK>']
    y_eos = y_char2idx['<EOS>']

    x_indices = [[x_char2idx.get(char, x_unk) for char in line] for line in x_data.split('\n')]
    y_indices = [[y_char2idx.get(char, y_unk) for char in line] + [y_eos] for line in y_data.split('\n')]

    return x_indices, y_indices, x_char2idx, y_char2idx, x_idx2char, y_idx2char


def main():
    batch_size = 128
    X_indices, Y_indices, X_char2idx, Y_char2idx, X_idx2char, Y_idx2char = preprocess_data()
    X_train = X_indices[batch_size:]
    Y_train = Y_indices[batch_size:]
    X_test = X_indices[:batch_size]
    Y_test = Y_indices[:batch_size]

    model = Seq2Seq(
        rnn_size=50,
        n_layers=2,
        X_word2idx=X_char2idx,
        encoder_embedding_dim=15,
        Y_word2idx=Y_char2idx,
        decoder_embedding_dim=15,
    )
    model.fit(X_train, Y_train, val_data=(X_test, Y_test), batch_size=batch_size)
    model.infer('common', X_idx2char, Y_idx2char)
    model.infer('apple', X_idx2char, Y_idx2char)
    model.infer('zhedong', X_idx2char, Y_idx2char)
# end function main


if __name__ == '__main__':
    main()
