#!/usr/bin/env python
# coding=utf-8

from seq2seq_attn import Seq2Seq

import sys
import argparse
import logging


if int(sys.version[0]) == 2:
    from io import open


def build_map(vocab_file):
    with open(vocab_file, 'r', encoding='utf8') as fin:
        words = list(set([line.strip() for line in fin]))

    idx2word = {idx: word for idx, word in enumerate(words)}
    word2idx = {word: idx for idx, word in idx2word.items()}
    return idx2word, word2idx


def read_data(path, word2idx, unk_id, eos_id=-1):
    with open(path, 'r', encoding='utf8') as fin:
        if eos_id == -1:
            return [[word2idx.get(word, unk_id) for word in line.strip().split()] for line in fin]
        else:
            return [[word2idx.get(word, unk_id) for word in line.strip().split()] + [eos_id] for line in fin]


def filter_pairs(x_data, y_data):
    x_new_data = []
    y_new_data = []

    max_p_len = 50
    for x_raw, y_raw in zip(x_data, y_data):
        if len(x_raw) > max_p_len or len(y_raw) > max_p_len:
            continue
        else:
            x_new_data.append(x_raw)
            y_new_data.append(y_raw)
    return x_new_data, y_new_data


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

    assert len(x_train) == len(y_train), 'numbers of pairs must be equal'
    assert len(x_valid_data) == len(y_valid_data), 'numbers of pairs must be equal'

    x_train, y_train = filter_pairs(x_train, y_train)
    x_valid_data, y_valid_data = filter_pairs(x_valid_data, y_valid_data)
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
    parser.add_argument("--predict", default=False, action='store_true')
    parser.add_argument("--model_path", default='', help="model path")

    # model settings
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--hidden_size", type=int, default=128, help="hidden size")
    parser.add_argument("--n_layers", type=int, default=2, help="number of layers in encoder")
    parser.add_argument("--n_epoch", type=int, default=10)

    return parser.parse_args()


def main(args):
    batch_size = args.batch_size
    print args

    # if args.predict:
    #    assert args.model_dir, 'model directory must be specified when predicting'

    x_train, y_train, x_valid, y_valid, x_word2idx, y_word2idx, x_idx2word, y_idx2word = \
        preprocess_data(args.input, args.validation, args.vocabulary)

    model = Seq2Seq(
        rnn_size=args.hidden_size,
        n_layers=args.n_layers,
        x_word2idx=x_word2idx,
        encoder_embedding_dim=args.hidden_size,
        y_word2idx=y_word2idx,
        decoder_embedding_dim=args.hidden_size,
        model_path=args.model_path
    )

    if not args.predict:
        model.build_graph()
        print 'Training ...'
        model.fit(x_train, y_train, val_data=(x_valid, y_valid), batch_size=batch_size, n_epoch=args.n_epoch)
    else:
        print 'Loading pre-trained model ...'
        model.restore_graph()

    print 'Translating ...'
    model.infer_sentence(u'我 的 青蛙 叫 呱呱 ！', x_idx2word, y_idx2word)
    model.infer_sentence(u'我 非常 期待 它 带 礼物 回来 !', x_idx2word, y_idx2word)

if __name__ == '__main__':
    main(parse_args())
