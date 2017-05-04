import tensorflow as tf
from utils import load_shakespeare_text
from rnn_text_gen import RNNTextGen


prime_texts = ['thou art more',
               'to be or not to',
               'wherefore art thou']


if __name__ == '__main__':
    text = load_shakespeare_text()

    sess = tf.Session()
    with tf.variable_scope('train_model'):
        train_model = RNNTextGen(sess, text)
    with tf.variable_scope('train_model', reuse=True):
        sample_model = RNNTextGen(sess, text=None, seq_len=1)
    log = train_model.fit(sample_model, prime_texts, text_iter_step=25)
    