from brown import get_indexed
from tfidf import Tfidf


if __name__ == '__main__':
    documents_indexed, word2idx = get_indexed(10000)
    vocab_size = len(word2idx)
    print("Data loaded | Vocab size:", vocab_size, '| Document size:', len(documents_indexed))

    model = Tfidf()
    TD = model.fit(documents_indexed, vocab_size)

    idx2word = {idx : word for word, idx in word2idx.items()}
    model.find_closest(['london', 'king', 'italy', 'queen'], TD, word2idx, idx2word)
