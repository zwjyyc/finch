import mxnet as mx


class RNNRegressor(mx.gluon.Block):
    def __init__(self, ctx, n_out, rnn_size=128, n_layer=1, lr=1e-3):
        super(RNNRegressor, self).__init__()
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
        rnn_out, hidden = self.lstm(x, hidden)
        logits = self.output(rnn_out.reshape((-1, self.rnn_size)))
        return logits, hidden
    # end method
        

    def compile_model(self):
        self.collect_params().initialize(mx.init.Xavier(), ctx=self.ctx)
        self.criterion = mx.gluon.loss.L2Loss()
        self.optimizer = mx.gluon.Trainer(self.collect_params(), 'adam', {'learning_rate': self.lr})
    # end method
# end class
