from config import args

import numpy as np


class DataLoader:
    def __init__(self, source_path, target_path):
        self.symbols = ['<pad>',  '<start>', '<end>', '<unknown>']

        self.source_words = self.read_data(source_path)
        self.target_words = self.read_data(target_path)

        self.source_word2idx = self.build_index(self.source_words)
        self.target_word2idx = self.build_index(self.target_words)

    
    def read_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()


    def build_index(self, data):
        chars = list(set([char for line in data.split('\n') for char in line]))
        return {char: idx for idx, char in enumerate(self.symbols + chars)}


    def pad(self, data, word2idx):
        res = []
        for line in data.split('\n'):
            temp_line = []
            for char in line:
                temp_line.append(word2idx.get(char, word2idx['<unknown>']))
            temp_line.append(word2idx['<end>'])
            if len(temp_line) > args.max_len:
                temp_line = temp_line[:args.max_len]
            if len(temp_line) < args.max_len:
                temp_line += [word2idx['<pad>']] * (args.max_len - len(temp_line))
            res.append(temp_line)
        return res


    def load(self):
        source_idx = self.pad(self.source_words, self.source_word2idx)
        target_idx = self.pad(self.target_words, self.target_word2idx)
        return np.array(source_idx), np.array(target_idx)
