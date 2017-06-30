"""
d: dictionary
k: key
v: value
l: list
"""
import string
import numpy as np


def remove_punct(s):
    return s.translate(None, string.punctuation)


def add2dict(d, k, v):
    if k not in d:
        d[k] = []
    d[k].append(v)


def list2proba_dict(l):
    d = {}
    for token in l:
        d[token] = d.get(token, 0) + 1
    for token, c in d.items():
        d[token] = float(c) / len(l)
    return d


def sample_word(d):
    probas = list(d.values())
    tokens = d.keys()
    idx = np.argmax(np.random.multinomial(1, probas, size=1)[0])
    return tokens[idx]


def main():
    first_words = {}
    second_words = {}
    transitions = {}

    for line in open('./temp/robert_frost.txt'):
        tokens = remove_punct(line.rstrip().lower()).split()
        
        for i, token in enumerate(tokens):
            if i == 0: # first word
                first_words[token] = first_words.get(token, 0) + 1
            else:
                if i == len(tokens) - 1: # last word
                    add2dict(transitions, (tokens[i-1], token), 'END')
                if i == 1: # second word
                    add2dict(second_words, tokens[i-1], token)
                else:
                    add2dict(transitions, (tokens[i-2], tokens[i-1]), token)

    total = sum(list(first_words.values()))
    for w, c in first_words.items():
        first_words[w] = float(c) / total

    for k, v in second_words.items():
        second_words[k] = list2proba_dict(v)

    for k, v in transitions.items():
        transitions[k] = list2proba_dict(v)

    for _ in range(4):
        sentence = []

        first_word = sample_word(first_words)
        sentence.append(first_word)

        second_word = sample_word(second_words[first_word])
        sentence.append(second_word)

        while True:
            next_word = sample_word(transitions[(first_word, second_word)])
            if next_word == 'END':
                break
            sentence.append(next_word)
            first_word = second_word
            second_word = next_word

        print(' '.join(sentence))


if __name__ == '__main__':
    main()
