import mxnet as mx
import numpy as np
import math


class CNNClassifier:
    def __init__(self, ctx, n_out, kernel_size=5, pool_size=2, lr=1e-3):
        self.ctx = ctx
        self.n_out = n_out
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.lr = lr
        self.build_model()
        self.compile_model()
    # end constructor


    def build_model(self):
        self.model = mx.gluon.nn.Sequential()
        self.model.add(mx.gluon.nn.Conv2D(32, self.kernel_size, padding=2))
        self.model.add(mx.gluon.nn.BatchNorm())
        self.model.add(mx.gluon.nn.MaxPool2D(self.pool_size))
        self.model.add(mx.gluon.nn.Activation(activation='relu'))

        self.model.add(mx.gluon.nn.Conv2D(64, self.kernel_size, padding=2))
        self.model.add(mx.gluon.nn.BatchNorm())
        self.model.add(mx.gluon.nn.MaxPool2D(self.pool_size))
        self.model.add(mx.gluon.nn.Activation(activation='relu'))

        self.model.add(mx.gluon.nn.Flatten())
        self.model.add(mx.gluon.nn.Dense(1024))
        self.model.add(mx.gluon.nn.BatchNorm())
        self.model.add(mx.gluon.nn.Activation(activation='relu'))

        self.model.add(mx.gluon.nn.Dense(self.n_out))
    # end method
        

    def compile_model(self):
        self.model.collect_params().initialize(mx.init.MSRAPrelu(), ctx=self.ctx)
        self.criterion = mx.gluon.loss.SoftmaxCrossEntropyLoss()
        self.optimizer = mx.gluon.Trainer(self.model.collect_params(), 'adam', {'learning_rate': self.lr})
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
                with mx.gluon.autograd.record(train_mode=True):
                    output = self.model(img)
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
            batch_pred = self.model(X_test_batch.as_in_context(self.ctx))
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
        optim = mx.gluon.Trainer(self.model.collect_params(), 'adam', {'learning_rate': lr})
        return optim, lr
    # end method
# end class
