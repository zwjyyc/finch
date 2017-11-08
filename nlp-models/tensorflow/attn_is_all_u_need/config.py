import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--max_len', type=int, default=20)
parser.add_argument('--hidden_units', type=int, default=128)
parser.add_argument('--num_blocks', type=int, default=6)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--dropout_rate', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--positional_encoding', type=str, default='learned')
parser.add_argument('--activation', type=str, default='elu')
parser.add_argument('--tied_proj_weight', type=int, default=1,
    help="to turn this on, you must explicitly assign 1")
parser.add_argument('--tied_embedding', type=int, default=0,
    help="to turn this on, you must explicitly assign 1")
parser.add_argument('--label_smoothing', type=int, default=1,
    help="to turn this on, you must explicitly assign 1")
parser.add_argument('--warmup_steps', type=int, default=4000)

args = parser.parse_args()