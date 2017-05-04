from keras.utils.data_utils import get_file
from rnn_text_gen import RNNTextGen
import tensorflow as tf


if __name__ == '__main__':
    path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
    text = open(path).read()
    
    sess = tf.Session()
    with tf.variable_scope('train_model'):
        train_model = RNNTextGen(sess, text)
    with tf.variable_scope('train_model', reuse=True):
        sample_model = RNNTextGen(sess, text=None, seq_len=1)
    log = train_model.fit(sample_model)
    