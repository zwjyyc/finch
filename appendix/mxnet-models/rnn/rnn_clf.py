import mxnet as mx
import numpy as np
import math


class RNNClassifier(mx.gluon.Block):
    def __init__(self, ctx, n_out, rnn_size=128, n_layer=1, lr=1e-3):
        super(RNNClassifier, self).__init__()
        self.ctx = ctx
        self.rnn_size = rnn_size
        self.n_layer = n_layer
        self.n_out = n_out
        self.lr = lr
        self.build_model()
        self.compile_model()
    # end constructor


    def build_model(self):
        self.lstm = mx.gluon.rnn.LSTM(self.rnn_size, self.n_layer, 'NTC')
        self.output = mx.gluon.nn.Dense(self.n_out)
    # end method


    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = out[:, out.shape[1]-1, :]
        out = self.output(out)
        return out
    # end method
        

    def compile_model(self):
        self.collect_params().initialize(mx.init.Xavier(), ctx=self.ctx)
        self.criterion = mx.gluon.loss.SoftmaxCrossEntropyLoss()
        self.optimizer = mx.gluon.Trainer(self.collect_params(), 'adam', {'learning_rate': self.lr})
    # end method


    def fit(self, X_train, y_train, batch_size=128, n_epoch=1):
        global_step = 0
        n_batch = len(X_train) // batch_size
        total_steps = n_epoch * n_batch

        X_train, y_train = self.from_numpy(X_train, y_train)
        train_loader = mx.gluon.data.DataLoader(mx.gluon.data.ArrayDataset(X_train, y_train),
                                                batch_size=batch_size, shuffle=True)
        for e in range(n_epoch):
            for i, (img, label) in enumerate(train_loader):
                img = img.as_in_context(self.ctx)
                label = label.as_in_context(self.ctx)
                h0 = mx.nd.zeros((self.n_layer, img.shape[0], 128), self.ctx)
                c0 = mx.nd.zeros((self.n_layer, img.shape[0], 128), self.ctx)
                with mx.gluon.autograd.record(train_mode=True):
                    output = self.forward(img, [h0, c0])
                    loss = self.criterion(output, label)
                self.optimizer, lr = self.adjust_lr(global_step, total_steps)
                loss.backward()
                self.optimizer.step(img.shape[0])

                global_step += 1
                loss = mx.nd.mean(loss).asscalar()
                preds = mx.nd.argmax(output, axis=1)
                acc = mx.nd.mean(preds == label).asscalar()
                if i % 50 == 0:
                    print('[{}/{}] [{}/{}] Loss: {:.4f}, Acc: {:.4f}, LR: {:.4f}'.format(
                          e+1, n_epoch, i, len(train_loader), loss, acc, lr))
    # end method


    def predict(self, X_test, batch_size=128):
        batch_pred_list = []
        test_loader = mx.gluon.data.DataLoader(self.from_numpy(X_test),
                                               batch_size=batch_size, shuffle=False)
        for X_test_batch in test_loader:
            h0 = mx.nd.zeros((self.n_layer, X_test_batch.shape[0], self.rnn_size), self.ctx)
            c0 = mx.nd.zeros((self.n_layer, X_test_batch.shape[0], self.rnn_size), self.ctx)
            batch_pred = self.forward(X_test_batch.as_in_context(self.ctx), [h0, c0])
            batch_pred_list.append(batch_pred.asnumpy())
        return np.argmax(np.vstack(batch_pred_list), 1)
    # end method


    def from_numpy(self, *args):
        data = []
        for _arr in args:
            arr = mx.nd.zeros(_arr.shape)
            arr[:] = _arr
            data.append(arr)
        return data if len(data) > 1 else data[0]
    # end method


    def adjust_lr(self, current_step, total_steps):
        max_lr = 3e-3
        min_lr = 1e-4
        decay_rate = math.log(min_lr/max_lr) / (-total_steps)
        lr = max_lr * math.exp(-decay_rate * current_step)
        optim = mx.gluon.Trainer(self.collect_params(), 'adam', {'learning_rate': lr})
        return optim, lr
    # end method
# end class
