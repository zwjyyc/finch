import re


def build_data():
    # B: 0, M: 1, E: 2, S: 3
    char2idx = {}
    char_idx = 0
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    def preprocess(path):
        text = open(path).read()
        text = text.replace('\n', ' ')
        text = re.sub('\s+', ' ', text)
        return text

    def build_y(chars, ys):
        if len(chars) == 1:
            ys.append(3)
        else:
            if i == 0:
                ys.append(0)
            elif i == len(chars) - 1:
                ys.append(2)
            else:
                ys.append(1)

    text_train = preprocess('temp/icwb2-data/training/pku_training.utf8')
    segs_train = text_train.split()
    text_test = preprocess('temp/icwb2-data/testing/pku_test.utf8')
    segs_test = text_test.split()

    for seg in segs_train:
        chars = list(seg.decode('utf-8'))
        for i, char in enumerate(chars):
            # handle x
            if char not in char2idx:
                char2idx[char] = char_idx
                char_idx += 1
            x_train.append(char2idx[char])
            # handle y
            build_y(chars, y_train)

    for seg in segs_test:
        chars = list(seg.decode('utf-8'))
        for i, char in enumerate(chars):
            # handle x
            if char in char2idx:
                x_test.append(char2idx[char])
            else:
                x_test.append(char_idx)
            # handle y
            build_y(chars, y_test)

    return x_train, y_train, x_test, y_test, len(char2idx), char2idx
