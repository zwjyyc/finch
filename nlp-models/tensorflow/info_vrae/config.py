import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--vocab_size', type=int, default=5000)
parser.add_argument('--num_sampled', type=int, default=500)
parser.add_argument('--max_len', type=int, default=30)
parser.add_argument('--word_dropout_rate', type=float, default=1.0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--rnn_cell', type=str, default='gru')
parser.add_argument('--rnn_size', type=int, default=128)
parser.add_argument('--encoder_layers', type=int, default=1)
parser.add_argument('--decoder_layers', type=int, default=1)
parser.add_argument('--beam_width', type=int, default=5)
parser.add_argument('--clip_norm', type=float, default=5.0)
parser.add_argument('--latent_size', type=int, default=16)
parser.add_argument('--anneal_max', type=float, default=1.0)
parser.add_argument('--anneal_bias', type=int, default=6000)
parser.add_argument('--mutinfo_loss', action='store_true')
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--display_loss_step', type=int, default=50)

args = parser.parse_args()
