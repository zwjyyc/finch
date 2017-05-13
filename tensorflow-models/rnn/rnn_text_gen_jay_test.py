from rnn_text_gen import RNNTextGen
import tensorflow as tf
import string


prime_texts = ['你要离开我知道很']


if __name__ == '__main__':
    with open('./temp/JayLyrics.txt', encoding='utf-8') as f:
        text = f.read()
    
    sess = tf.Session()
    model = RNNTextGen(sess, text, n_layer=3, min_freq=20)
    log = model.fit_text(prime_texts, text_iter_step=1)
