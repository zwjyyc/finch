from rnn_text_gen import RNNTextGen
import tensorflow as tf
import string


stopwords = [x for x in string.punctuation if x not in ['-', "'"]]
prime_texts = ['I']


if __name__ == '__main__':
    path = tf.contrib.keras.utils.get_file('nietzsche.txt',
                                            origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
    text = open(path).read()
    model = RNNTextGen(text, stopwords=stopwords)
    log = model.fit_text(prime_texts, text_iter_step=3)
