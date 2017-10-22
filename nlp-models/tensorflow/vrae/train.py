from data import IMDB
from model import VRAE
from config import args
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
                bar = '[%d/%d] | [%d/%d] | nll_loss: %.1f | kl_w: %.3f | kl_loss: %.1f | mutinfo_loss: %.1f'
                vars = (epoch+1, args.num_epoch, i+1, len(dataloader._X)//args.batch_size, log['nll_loss'],
                        log['kl_w'], log['kl_loss'], log['mutinfo_loss'])
                print(bar % vars)
            if i % args.display_text_step == 0:
                model.reconstruct(sess, seq[-1])
                model.generate(sess)


if __name__ == '__main__':
    main()