from conv_rnn_char import ConvLSTMChar
import tensorflow as tf
import string


stopwords = [x for x in string.punctuation if x not in ['-', "'"]]


if __name__ == '__main__':
    path = tf.contrib.keras.utils.get_file('nietzsche.txt',
                                            origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
    text = open(path).read()
    
    model = ConvLSTMChar(text, stopwords=stopwords, min_freq=5)
    log = model.fit_text(text_iter_step=3)
