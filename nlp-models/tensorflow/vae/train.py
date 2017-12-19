from __future__ import print_function
from data import IMDB
from model import VRAE
from config import args
import json
import tensorflow as tf


def main():
    dataloader = IMDB()
    model = VRAE(dataloader.word2idx, dataloader.idx2word)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(args.num_epoch):
        dataloader.update_word_dropout()
        print("\nWord Dropout")
        dataloader.shuffle()
        print("Data Shuffled", end='\n\n')
        for i, (enc_inp, dec_inp, dec_out) in enumerate(dataloader.next_batch()):
            log = model.train_session(sess, enc_inp, dec_inp, dec_out)
            if i % args.display_loss_step == 0:
                print("Step %d | [%d/%d] | [%d/%d]" % (log['step'], epoch+1, args.num_epoch, i, len(dataloader.enc_inp)//args.batch_size), end='')
                print(" | nll_loss:%.1f | kl_w:%.3f | kl_loss:%.2f \n" % (log['nll_loss'], log['kl_w'], log['kl_loss']))
        
        model.reconstruct(sess, enc_inp[-1], dec_inp[-1])
        #model.generate(sess)
        model.customized_reconstruct(sess, 'i love this film it is one of the best i want to watch again')
        model.customized_reconstruct(sess, 'this movie is a waste of time and there is no point to see it')
        
        save_path = model.saver.save(sess, model.model_path)
        print("Model saved in file: %s" % save_path)


if __name__ == '__main__':
    print(json.dumps(args.__dict__, indent=4))
    main()