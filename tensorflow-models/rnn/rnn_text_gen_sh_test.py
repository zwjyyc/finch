import os
import requests
import string
from rnn_text_gen import RNNTextGen


stopwords = [x for x in string.punctuation if x not in ['-', "'"]]
prime_texts = ['I']


def load_shakespeare_text():
    data_dir = 'temp'
    data_file = 'shakespeare.txt'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    print('Loading Shakespeare Data')
    if not os.path.isfile(os.path.join(data_dir, data_file)):   # check if file is downloaded
        print('Not found, downloading Shakespeare texts from www.gutenberg.org')
        shakespeare_url = 'http://www.gutenberg.org/cache/epub/100/pg100.txt'
        response = requests.get(shakespeare_url)
        shakespeare_file = response.content
        s_text = shakespeare_file.decode('utf-8')               # decode binary -> string
        s_text = s_text[7675:]                                  # drop first few descriptive paragraphs
        with open(os.path.join(data_dir, data_file), 'w') as f: # write to file
            f.write(s_text)
    else:
        with open(os.path.join(data_dir, data_file), 'r') as f: # if file has been saved, load from that file
            s_text = f.read()
    return s_text
# end function load_shakespeare_text


if __name__ == '__main__':
    text = load_shakespeare_text()
    model = RNNTextGen(text, stopwords=stopwords)
    log = model.fit_text(prime_texts, text_iter_step=25)
