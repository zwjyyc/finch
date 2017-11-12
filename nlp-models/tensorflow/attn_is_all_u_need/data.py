from config import args
from collections import Counter

import numpy as np


class DataLoader:
    def __init__(self, source_path, target_path):
        self.source_words = self.read_data(source_path)
        self.target_words = self.read_data(target_path)

        self.source_word2idx = self.build_index(self.source_words)
        self.target_word2idx = self.build_index(self.target_words, is_target=True)

    
    def read_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()


    def build_index(self, data, is_target=False):
        chars = [char for line in data.split('\n') for char in line]
        most_common = Counter(chars).most_common(args.vocab_size)
        chars = [char for char, freq in most_common]
        if is_target:
            symbols = ['<pad>','<start>','<end>','<unk>']
            return {char: idx for idx, char in enumerate(symbols + chars)}
        else:
            symbols = ['<pad>','<unk>'] if args.tied_embedding==0 else  ['<pad>','<start>','<end>','<unk>']
            return {char: idx for idx, char in enumerate(symbols + chars)}


    def pad(self, data, word2idx, is_target=False):
        res = []
        for line in data.split('\n'):
            temp_line = [word2idx.get(char, word2idx['<unk>']) for char in line]
            if is_target:
                temp_line.append(word2idx['<end>'])
            if len(temp_line) > args.max_len:
                temp_line = temp_line[:args.max_len]
            if len(temp_line) < args.max_len:
                temp_line += [word2idx['<pad>']] * (args.max_len - len(temp_line))
            res.append(temp_line)
        return np.array(res)


    def load(self):
        source_idx = self.pad(self.source_words, self.source_word2idx)
        target_idx = self.pad(self.target_words, self.target_word2idx, is_target=True)
        return source_idx, target_idx
