import numpy as np
from rnn_regr import RNNRegressor


BATCH_START = 0     
TIME_STEPS = 20     
BATCH_SIZE = 50  


def get_data_batch(return_x=False):
    global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)
    xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS*BATCH_SIZE
    # returned seq, res: shape (batch, step, input)
    if return_x:
        [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]
    else:
        return (seq[:, :, np.newaxis], res[:, :, np.newaxis])


if __name__ == '__main__':
    data = [get_data_batch() for _ in range(2000)]
    train_data = data[:1500]
    test_data = data[1500:]
    regr  = RNNRegressor(n_in=1, n_step=TIME_STEPS, n_hidden=16, n_out=1)
    regr.fit(train_data, BATCH_SIZE, test_data)
