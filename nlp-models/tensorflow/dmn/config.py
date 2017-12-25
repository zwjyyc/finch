import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--num_epoch', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--embed_dim', type=int, default=80)
parser.add_argument('--hidden_size', type=int, default=80)
parser.add_argument('--dropout_rate', type=float, default=0.1)
parser.add_argument('--n_episodes', type=int, default=3)
parser.add_argument('--clip_norm', type=float, default=5.0)
parser.add_argument('--display_step', type=int, default=50)

args = parser.parse_args()