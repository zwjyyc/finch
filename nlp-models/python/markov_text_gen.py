from __future__ import division
import sys
import string
from io import open


def remove_punct(s):
    if int(sys.version[0]) == 2:
        return s.translate(None, string.punctuation)
    else:
        table = str.maketrans({p: '' for p in string.punctuation})
        return s.translate(table)
# end function remove_punct


# d: dictionary, k: key, v: value, l: list
def add2dict(d, k, v):
    if k not in d:
        d[k] = []
    d[k].append(v)
# end function add2dict


def list2proba_dict(l):
    d = {}
    for token in l:
        d[token] = d.get(token, 0) + 1
    for token, c in d.items():
        d[token] = c / len(l)
    return d
# end function list2proba_dict


def build_model(f_path):
    first_words = {}
    second_words = {}
    transitions = {}

    for line in open(f_path, encoding='utf-8'):
        tokens = remove_punct(line.rstrip().lower()).split()
        
        for i, token in enumerate(tokens):
            if i == 0:                   # first word
                first_words[token] = first_words.get(token, 0) + 1
            else:
                if i == len(tokens) - 1: # last word
                    add2dict(transitions, (tokens[i-1], token), 'END')
                if i == 1:               # second word
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

    return first_words, second_words, transitions
# end function build_model
