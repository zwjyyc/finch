import random
import numpy as np
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize


trigram = {}
trigram2proba = {}
replace_rate = 0.2


def predict(token2proba):
    probas = list(token2proba.values())
    tokens = token2proba.keys()
    idx = np.argmax(np.random.multinomial(1, probas, size=1)[0])
    return tokens[idx]



if __name__ == '__main__':
    pos_reviews = BeautifulSoup(open('temp/positive.review').read(), 'lxml').findAll('review_text')

    for review in pos_reviews:
        string = review.text.lower()
        tokens = word_tokenize(string)
        for i in range(len(tokens) - 2):
            key = (tokens[i], tokens[i+2])
            if key not in trigram:
                trigram[key] = []
            trigram[key].append(tokens[i+1])

    for key, words in trigram.iteritems():
        if len(set(words)) > 1:
            word2proba = {}
            for word in words:
                if word not in word2proba:
                    word2proba[word] = 1
                word2proba[word] += 1
            total_count = sum(list(word2proba.values()))
            for word, count in word2proba.iteritems():
                word2proba[word] = float(count) / total_count
            trigram2proba[key] = word2proba

    review = random.choice(pos_reviews)
    string = review.text.lower().strip()
    print(string)
    tokens = word_tokenize(string)
    for i in range(len(tokens) - 2):
        if random.random() < replace_rate:
            key = (tokens[i], tokens[i+2])
            if key in trigram2proba:
                next_word = predict(trigram2proba[key])
                tokens[i+1] = next_word
    print()
    print(' '.join(tokens).replace(' .', '.').replace(" '", "'").replace(' ,', ',').replace(' $', '$').replace(' !', '!'))
