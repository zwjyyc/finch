from __future__ import print_function
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from trigram import Trigram
import random


replace_rate = 0.2


if __name__ == '__main__':
    reviews = BeautifulSoup(open('temp/positive.review').read(), 'lxml').findAll('review_text')
    documents = [review.text for review in reviews]
    print("Data Loaded")

    model = Trigram()
    model.fit(documents)

    while True:
        review = random.choice(documents)
        if len(review.split()) < 30:
            break
    string = review.lower().strip()
    print(string, end='\n\n')

    tokens = word_tokenize(string)
    for i in range(len(tokens) - 2):
        if random.random() < replace_rate:
            key = (tokens[i], tokens[i+2])
            if key in model.trigram2proba:
                next_word = model.predict(key)
                tokens[i+1] = next_word
    print(' '.join(tokens).replace(' .', '.').replace(" '", "'").replace(' ,', ',').replace(' $', '$').replace(' !', '!'))
    