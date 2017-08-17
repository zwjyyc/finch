import pos
import numpy as np
from birnn_seq_clf import BiRNN


SEQ_LEN = 20
BATCH_SIZE = 32
N_EPOCH = 5
sample = ['I', 'love', 'you']


def to_seq(*args):
    data = []
    for x in args:
        x = x[: (len(x) - len(x) % SEQ_LEN)]
        data.append(np.reshape(x, [-1, SEQ_LEN]))
    return data


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, vocab_size, n_class, word2idx, tag2idx = pos.load_data()
    X_train, X_test, Y_train, Y_test = to_seq(x_train, x_test, y_train, y_test)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    clf = BiRNN(vocab_size, n_class)
    clf.fit(X_train, Y_train, n_epoch=N_EPOCH, batch_size=BATCH_SIZE)
    clf.evaluate(X_test, Y_test)

    preds = clf.infer([word2idx[w] for w in sample])
    indices = np.argmax(preds, 1)
    idx2tag = {idx : tag for tag, idx in tag2idx.items()}
    print(' '.join(sample))
    print(' '.join([idx2tag[idx] for idx in indices]))
