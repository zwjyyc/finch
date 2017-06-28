from rnn_regr import RNNRegressor
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


TIME_STEPS = 20
BATCH_SIZE = 50


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


if __name__ == '__main__':
    train_gen = TimeSeriesGen(0, TIME_STEPS, BATCH_SIZE)
    test_gen = TimeSeriesGen(1000, TIME_STEPS, BATCH_SIZE)
    model = RNNRegressor(n_in = 1,
                         n_out = 1,
                         cell_size = TIME_STEPS)

    train_state = None
    test_state = None
    sns.set(style='white')
    plt.ion()

    for step in range(200):
        X_train, Y_train, _ = train_gen.next_batch()
        X_train = torch.autograd.Variable(torch.from_numpy(X_train.astype(np.float32)))
        Y_train = torch.autograd.Variable(torch.from_numpy(Y_train.astype(np.float32)))

        Y_pred, train_state = model.forward(X_train, train_state)
        train_state = (torch.autograd.Variable(train_state.data))

        loss = model.criterion(Y_pred, Y_train)    
        model.optimizer.zero_grad()                             
        loss.backward()                                        
        model.optimizer.step()

        X_test, Y_test, ts = test_gen.next_batch()
        X_test = torch.autograd.Variable(torch.from_numpy(X_test.astype(np.float32)))
        
        Y_pred, test_state = model.forward(X_test, test_state)
        test_state = (torch.autograd.Variable(test_state.data))
        Y_pred = Y_pred.data.numpy()

        # update plotting
        plt.cla()
        plt.plot(ts.ravel(), Y_test.ravel(), label='actual')
        plt.plot(ts.ravel(), Y_pred.ravel(), color='indianred', label='predicted')
        plt.ylim((-1.2, 1.2))
        plt.xlim((ts.ravel()[0], ts.ravel()[-1]))
        plt.legend(fontsize=15)
        plt.draw()
        plt.pause(0.01)
        print('train loss: %.4f' % (loss.data.numpy()))

    plt.ioff()
    plt.show()
