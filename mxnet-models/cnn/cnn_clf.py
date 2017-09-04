import mxnet as mx


class CNNClassifier:
    def __init__(self, n_out, lr=1e-3):
        self.n_out = n_out
        self.lr = lr
        self.build_model()
        self.compile_model()
    # end constructor


    def build_model(self):
        self.model = mx.gluon.nn.Sequential()
        self.model.add(mx.gluon.nn.Conv2D(16, 5, padding=2, activation='relu'))
        self.model.add(mx.gluon.nn.MaxPool2D(2))
        self.model.add(mx.gluon.nn.Conv2D(32, 5, padding=2, activation='relu'))
        self.model.add(mx.gluon.nn.MaxPool2D(2))
        self.model.add(mx.gluon.nn.Flatten())
        self.model.add(mx.gluon.nn.Dense(10))
    # end method
        

    def compile_model(self):
        self.model.collect_params().initialize(mx.init.Xavier())
        self.criterion = mx.gluon.loss.SoftmaxCrossEntropyLoss()
        self.optimizer = mx.gluon.Trainer(self.model.collect_params(), 'adam', {'learning_rate': self.lr})
    # end method


    def fit(self, X_train, y_train, batch_size=128, n_epoch=1):
        X_train, y_train = self.from_numpy(X_train, y_train)
        train_loader = mx.gluon.data.DataLoader(mx.gluon.data.ArrayDataset(X_train, y_train),
                                                batch_size=batch_size, shuffle=True)
        for e in range(n_epoch):
            for i, (img, label) in enumerate(train_loader):
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


    def from_numpy(self, *args):
        data = []
        for _arr in args:
            arr = mx.nd.zeros(_arr.shape)
            arr[:] = _arr
            data.append(arr)
        return data
    # end method
# end class
