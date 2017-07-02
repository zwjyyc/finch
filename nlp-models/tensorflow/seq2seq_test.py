def read_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
# end function read_data


def build_map(data):
    specials = ['<PAD>', '<UNK>', '<GO>',  '<EOS>']
    chars = list(set([char for line in data.split('\n') for char in line]))
    idx2char = {idx: char for idx, char in enumerate(specials + chars)}
    char2idx = {char: idx for idx, char in idx2char.items()}
    return idx2char, char2idx
# end function build_map


def preprocess_data()
    X_data = read_data('temp/letters_source.txt')
    Y_data = read_data('temp/letters_target.txt')

    X_idx2char, X_char2idx = build_map(X_data)
    Y_idx2char, Y_char2idx = build_map(Y_data)

    X_indices = [[X_char2idx.get(char, X_char2idx['<UNK>']) for char in line] for line in X_data.split('\n')]
    Y_indices = [[Y_char2idx.get(char, Y_char2idx['<UNK>']) for char in line] + [Y_char2idx['<EOS>']]
                  for line in Y_data.split('\n')]
# end function preprocess_data


def main():
    preprocess_data()
# end function main

if __name__ == '__main__':
    main()
