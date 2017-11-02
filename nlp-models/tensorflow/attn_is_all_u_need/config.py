import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--max_len', type=int, default=20)
parser.add_argument('--hidden_units', type=int, default=512)
parser.add_argument('--num_blocks', type=int, default=6)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--dropout_rate', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=20)

args = parser.parse_args()