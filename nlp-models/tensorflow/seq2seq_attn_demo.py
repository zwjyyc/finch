#!/usr/bin/env python
# coding=utf-8

from seq2seq_attn import Seq2Seq

import sys
import argparse
import logging


if int(sys.version[0]) == 2:
    from io import open


def build_map(vocab_file):
    with open(vocab_file, 'r') as fin:
        words = list(set([line.decode('utf8').strip() for line in fin]))

    idx2word = {idx: word for idx, word in enumerate(words)}
    word2idx = {word: idx for idx, word in idx2word.items()}
    return idx2word, word2idx


def read_data(path, word2idx, unk_id, eos_id=-1):
    with open(path, 'r', encoding='utf-8') as fin:
        if eos_id == -1:
            return [[word2idx.get(word, unk_id)] for line in fin for word in line.strip().split()]
        else:
            return [[word2idx.get(word, unk_id)] + [eos_id] for line in fin for word in line.strip().split()]


def preprocess_data(src_files, valid_files, vocab_files):
    x_idx2word, x_word2idx = build_map(vocab_files[0])
    y_idx2word, y_word2idx = build_map(vocab_files[1])

    out_str = 'Have obtained {} words from {};\nHave obtained {} words from {}'.format(
        len(x_word2idx), vocab_files[0], len(y_word2idx), vocab_files[1])
    print out_str

    x_unk = x_word2idx['<unk>']
    y_unk = y_word2idx['<unk>']
    y_eos = y_word2idx['<eos>']

    x_train = read_data(src_files[0], x_word2idx, x_unk)
    y_train = read_data(src_files[1], y_word2idx, y_unk, y_eos)
    x_valid_data = read_data(valid_files[0], x_word2idx, x_unk)
    y_valid_data = read_data(valid_files[1], y_word2idx, y_unk, y_eos)

    print x_train

    print y_train

    return x_train, y_train, x_valid_data, y_valid_data, x_word2idx, y_word2idx, x_idx2word, y_idx2word


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training neural machine translation models",
        usage="trainer.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--input", type=str, nargs=2,
                        help="Path of source and target corpus")
    parser.add_argument("--output", type=str, default="train",
                        help="Path to saved models")
    parser.add_argument("--vocabulary", type=str, nargs=2,
                        help="Path of source and target vocabulary")
    parser.add_argument("--validation", type=str, nargs=2,
                        help="Path of validation file")

    # model settings
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--hidden_size", type=int, default=128, help="hidden size")
    parser.add_argument("--n_layers", type=int, default=2, help="number of layers in encoder")

    return parser.parse_args()


def main(args):
    batch_size = args.batch_size

    x_train, y_train, x_valid, y_valid, x_word2idx, y_word2idx, x_idx2word, y_idx2word = preprocess_data(args.input, args.validation, args.vocabulary)

    model = Seq2Seq(
        rnn_size=args.hidden_size,
        n_layers=args.n_layers,
        X_word2idx=x_word2idx,
        encoder_embedding_dim=args.hidden_size,
        Y_word2idx=y_word2idx,
        decoder_embedding_dim=args.hidden_size,
    )

    model.fit(x_train, y_train, val_data=(x_valid, y_valid), batch_size=batch_size)
    model.infer(u'我的青蛙叫呱呱！', x_idx2word, y_idx2word)

if __name__ == '__main__':
    main(parse_args())
