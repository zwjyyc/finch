import re
import sys
import jieba


def load_data():
    # B: 0, M: 1, E: 2, S: 3
    char2idx = {}
    idx2char = {}
    char_idx = 0
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    version = int(sys.version[0])

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
    
    if version == 3:
        text_train = preprocess('temp/icwb2-data/training/pku_training.txt')
        text_train += preprocess('temp/icwb2-data/training/msr_training.txt')
    else:
        text_train = preprocess('temp/icwb2-data/training/pku_training.utf8')
        text_train += preprocess('temp/icwb2-data/training/msr_training.utf8')
    segs_train = text_train.split()
    if version == 3:
        text_test = preprocess('temp/icwb2-data/testing/pku_test.txt')
        text_test += preprocess('temp/icwb2-data/testing/msr_test.txt')
    else:
        text_test = preprocess('temp/icwb2-data/testing/pku_test.utf8')
        text_test += preprocess('temp/icwb2-data/testing/msr_test.utf8')
    segs_test = jieba.cut(text_test)

    for seg in segs_train:
        chars = list(seg) if version == 3 else list(seg.decode('utf-8'))
        for i, char in enumerate(chars):
            # handle x
            if char not in char2idx:
                char2idx[char] = char_idx
                idx2char[char_idx] = char
                char_idx += 1
            x_train.append(char2idx[char])
            # handle y
            build_y(chars, y_train)

    char2idx['_unknown'] = char_idx

    for seg in segs_test:
        chars = list(seg) if version == 3 else list(seg.decode('utf-8'))
        for i, char in enumerate(chars):
            # handle x
            if char in char2idx:
                x_test.append(char2idx[char])
            else:
                x_test.append(char_idx)
            # handle y
            build_y(chars, y_test)
    
    return x_train, y_train, x_test, y_test, len(char2idx), char2idx, idx2char
