import tensorflow as tf
from utils import load_shakespeare_text
from rnn_text_gen import RNNTextGen


prime_texts = ['thou art more',
               'to be or not to',
               'wherefore art thou']


if __name__ == '__main__':
    text = load_shakespeare_text()

    sess = tf.Session()
    train_model = RNNTextGen(sess, text)
    log = train_model.learn()
    