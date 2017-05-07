from keras.utils.data_utils import get_file
from conv_rnn_text_gen import RNNTextGen
import tensorflow as tf


if __name__ == '__main__':
    path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
    text = open(path).read()
    
    sess = tf.Session()
    train_model = RNNTextGen(sess, text)
    log = train_model.learn()
    