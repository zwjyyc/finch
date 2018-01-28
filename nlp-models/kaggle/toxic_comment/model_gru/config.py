import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--vocab_size', type=int, default=20000)
parser.add_argument('--embed_dim', type=int, default=300)

parser.add_argument('--max_len', type=int, default=100)
parser.add_argument('--rnn_size', type=int, default=100)
parser.add_argument('--clip_norm', type=float, default=5.0)

parser.add_argument('--num_epochs', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--test_size', type=float, default=0.1)

args = parser.parse_args()
