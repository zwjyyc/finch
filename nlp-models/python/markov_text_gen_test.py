from markov_text_gen import build_model
import numpy as np


def sample_word(d):
    probas = list(d.values())
    tokens = list(d.keys())
    idx = np.argmax(np.random.multinomial(1, probas, size=1)[0])
    return tokens[idx]
# end function sample_word


def generate(first_words, second_words, transitions):
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
# end function generate


def main():
    generate(*build_model('./temp/robert_frost.txt'))


if __name__ == '__main__':
    main()