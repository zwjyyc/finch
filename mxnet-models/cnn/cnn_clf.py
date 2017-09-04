import mxnet as mx
import numpy as np


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

        self.model.add(mx.gluon.nn.Dense(10))
    # end method
        

    def compile_model(self):
        self.model.collect_params().initialize(mx.init.Xavier(), ctx=self.ctx)
        self.criterion = mx.gluon.loss.SoftmaxCrossEntropyLoss()
        self.optimizer = mx.gluon.Trainer(self.model.collect_params(), 'adam', {'learning_rate': self.lr})
    # end method


    def fit(self, X_train, y_train, batch_size=128, n_epoch=1):
        X_train, y_train = self.from_numpy(X_train, y_train)
        train_loader = mx.gluon.data.DataLoader(mx.gluon.data.ArrayDataset(X_train, y_train),
                                                batch_size=batch_size, shuffle=True)
        for e in range(n_epoch):
            for i, (img, label) in enumerate(train_loader):
                img = img.as_in_context(self.ctx)
                label = label.as_in_context(self.ctx)
                with mx.gluon.autograd.record():
                    output = self.model(img)
                    loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step(img.shape[0])
                loss = mx.nd.mean(loss).asscalar()
                predict = mx.nd.argmax(output, axis=1)
                acc = mx.nd.mean(predict == label).asscalar()
                if i % 50 == 0:
                    print('[{}/{}] [{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(e+1, n_epoch, i, len(train_loader), loss, acc))
    # end method


    def predict(self, X_test, batch_size=128):
        batch_pred_list = []
        for X_test_batch in self.gen_batch(X_test, batch_size):
            batch_pred = self.model(self.from_numpy(X_test_batch)[0].as_in_context(self.ctx))
            batch_pred_list.append(batch_pred.asnumpy())
        return np.argmax(np.vstack(batch_pred_list), 1)
    # end method


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method


    def from_numpy(self, *args):
        data = []
        for _arr in args:
            arr = mx.nd.zeros(_arr.shape)
            arr[:] = _arr
            data.append(arr)
        return data
    # end method
# end class
