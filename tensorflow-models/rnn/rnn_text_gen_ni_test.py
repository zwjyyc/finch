from keras.utils.data_utils import get_file
from rnn_text_gen import RNNTextGen
import tensorflow as tf
import string

stopwords = [x for x in string.punctuation if x not in ['-', "'"]]

if __name__ == '__main__':
    path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
    text = open(path).read()
    
    sess = tf.Session()
    train_model = RNNTextGen(sess, text, stopwords=stopwords)
    log = train_model.learn()
    