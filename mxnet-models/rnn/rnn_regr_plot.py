from rnn_regr import RNNRegressor
import mxnet as mx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


TIME_STEPS = 20
BATCH_SIZE = 50
N_STEPS = 200


class TimeSeriesGen:
    def __init__(self, batch_start, time_steps, batch_size):
        self.batch_start = batch_start
        self.time_steps = time_steps
        self.batch_size = batch_size
        
    def next_batch(self):
        ts = np.arange(self.batch_start, self.batch_start + self.time_steps * self.batch_size)
        ts = ts.reshape([self.batch_size, self.time_steps]) / (10 * np.pi)
        self.batch_start += self.time_steps * self.batch_size
        X = np.sin(ts)
        Y = np.cos(ts)
        return X[:, :, np.newaxis], Y[:, :, np.newaxis], ts
# end class


def from_numpy(*args):
    data = []
    for _arr in args:
        arr = mx.nd.zeros(_arr.shape)
        arr[:] = _arr
        data.append(arr)
    return data if len(data) > 1 else data[0]
# end function


def detach(arrs):
    return [arr.detach() for arr in arrs]
# end function


def main():
    train_gen = TimeSeriesGen(0, TIME_STEPS, BATCH_SIZE)
    test_gen = TimeSeriesGen(1000, TIME_STEPS, BATCH_SIZE)
    model = RNNRegressor(mx.cpu(), n_out = 1, rnn_size = TIME_STEPS)

    init_state = mx.nd.zeros((1, BATCH_SIZE, TIME_STEPS), model.ctx)
    train_state = [init_state] * 2
    test_state = [init_state] * 2
    sns.set(style='white')
    plt.ion()

    for step in range(N_STEPS):
        X_train, Y_train, _ = train_gen.next_batch()
        X_train, Y_train = from_numpy(X_train, Y_train)
        with mx.gluon.autograd.record(train_mode=True):
            Y_pred, train_state = model(X_train, train_state)
            train_state = detach(train_state)
            loss = model.criterion(Y_pred, Y_train)                             
        loss.backward()                                        
        model.optimizer.step(BATCH_SIZE)

        X_test, Y_test, ts = test_gen.next_batch()
        X_test = from_numpy(X_test)
        Y_pred, test_state = model(X_test, test_state)
        Y_pred = Y_pred.asnumpy()

        plt.cla()
        plt.plot(ts.ravel(), Y_test.ravel(), label='actual')
        plt.plot(ts.ravel(), Y_pred.ravel(), color='indianred', label='predicted')
        plt.ylim((-1.2, 1.2))
        plt.xlim((ts.ravel()[0], ts.ravel()[-1]))
        plt.legend(fontsize=15, loc=1)
        plt.draw()
        plt.pause(5e-3)
        print('[%d/%d] train loss: %.4f' % (step, N_STEPS, mx.nd.mean(loss).asscalar()))

    plt.ioff()
    plt.show() 
# end function


if __name__ == '__main__':
    main()
