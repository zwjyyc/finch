from data import IMDB
from model import VRAE
from config import args
import json
import tensorflow as tf


def main():
    dataloader = IMDB()
    model = VRAE(dataloader.word2idx)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(args.num_epoch):
        dataloader.update_word_dropout()
        dataloader.shuffle()
        print("\nData Shuffled", end='\n\n')
        for i, (seq, seq_dropped, seq_len) in enumerate(dataloader.next_batch()):
            log = model.train_session(sess, seq, seq_dropped, seq_len)
            if i % args.display_loss_step == 0:
                print("Step %d | [%d/%d] | [%d/%d]" % (log['step'], epoch+1, args.num_epoch, i,
                    len(dataloader._X)//args.batch_size), end='')
                if args.mutinfo_loss:
                    print("\n\tnll_loss:%.1f | kl_w:%.3f | kl_loss:%.2f | temper:%.2f | mutinfo_loss:%.2f \n" % (
                        log['nll_loss'], log['kl_w'], log['kl_loss'], log['temperature'], log['mutinfo_loss']))
                else:
                    print(" | nll_loss:%.1f | kl_w:%.3f | kl_loss:%.2f \n" % (
                        log['nll_loss'], log['kl_w'], log['kl_loss']))
        model.reconstruct(sess, seq[-1], seq_dropped[-1])
        model.generate(sess)


if __name__ == '__main__':
    print(json.dumps(args.__dict__, indent=4))
    main()