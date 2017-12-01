from __future__ import print_function
from data import IMDB
from model import VRAE
import tensorflow as tf


def main():
    dataloader = IMDB()
    model = VRAE(dataloader.word2idx)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print("Loading trained model ...")
    model.saver.restore(sess, model.model_path)

    # lowercase, no punctuation, please 
    model.customized_reconstruct(sess, 'i love this firm it is one of the best')
    model.customized_reconstruct(sess, 'i want to see this movie it seems interesting')
    

if __name__ == '__main__':
    main()