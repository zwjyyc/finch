import mxnet as mx
import numpy as np


class Generator(mx.gluon.Block):
    def __init__(self, G_size, img_ch, shape_trace, kernel_size, stride, padding):
        super(Generator, self).__init__()
        self.G_size = G_size
        self.img_ch = img_ch
        self.shape_trace = shape_trace
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.build_model()
    # end constructor


    def build_model(self):
        # for example: 100 -> (7, 7, 128) ->  (14, 14, 64) -> (28, 28, 1)
        self.linear = mx.gluon.nn.Dense(np.prod(self.shape_trace[0]))
        self.lrelu = mx.gluon.nn.LeakyReLU(0.2)
        self.deconv = mx.gluon.nn.Sequential()
        for shape in (self.shape_trace[1:]):
            self.deconv.add(mx.gluon.nn.Conv2DTranspose(shape[-1], self.kernel_size, self.stride, self.padding))
            self.deconv.add(mx.gluon.nn.BatchNorm())
            self.deconv.add(mx.gluon.nn.LeakyReLU(0.2))
        self.deconv.add(mx.gluon.nn.Conv2DTranspose(self.img_ch, self.kernel_size, self.stride, self.padding))
        self.deconv.add(mx.gluon.nn.Activation('tanh'))
    # end method


    def forward(self, x):
        x = self.lrelu(self.linear(x))
        X = x.reshape([-1, self.shape_trace[0][2], self.shape_trace[0][0], self.shape_trace[0][1]])
        return self.deconv(X)           
    # end method
# end class


class Discriminator(mx.gluon.Block):
    def __init__(self, img_size, img_ch, shape_trace, kernel_size, stride, padding):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.img_ch = img_ch
        self.shape_trace = shape_trace
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.build_model()
    # end constructor


    def build_model(self):
        # for example: (28, 28, 1) -> (14, 14, 64) -> (7, 7, 128) -> 1
        self.model = mx.gluon.nn.Sequential()
        for shape in list(reversed(self.shape_trace)):
            self.model.add(mx.gluon.nn.Conv2D(shape[-1], self.kernel_size, self.stride, self.padding))
            self.model.add(mx.gluon.nn.BatchNorm())
            self.model.add(mx.gluon.nn.LeakyReLU(0.2))
        self.model.add(mx.gluon.nn.Flatten())
        self.model.add(mx.gluon.nn.Dense(2))
    # end method


    def forward(self, X):
        return self.model(X)     
    # end method
# end class


class GAN:
    def __init__(self, ctx, G_size, img_size, img_ch, shape_trace, kernel_size=4, stride=2, padding=1):
        self.ctx = ctx
        self.G_size = G_size
        self.img_size = img_size
        self.img_ch = img_ch
        self.shape_trace = shape_trace
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.build_model()
    # end constructor


    def build_model(self):
        self.g = Generator(self.G_size, self.img_ch, self.shape_trace, self.kernel_size, self.stride, self.padding)
        self.d = Discriminator(self.img_size, self.img_ch, self.shape_trace, self.kernel_size, self.stride, self.padding)
        self.g.collect_params().initialize(mx.init.MSRAPrelu(), ctx=self.ctx)
        self.d.collect_params().initialize(mx.init.MSRAPrelu(), ctx=self.ctx)
        self.g_optim = mx.gluon.Trainer(self.g.collect_params(), 'adam', {'learning_rate':2e-4, 'beta1':0.5})
        self.d_optim = mx.gluon.Trainer(self.d.collect_params(), 'adam', {'learning_rate':2e-4, 'beta1':0.5})
        self.loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
        self.mse = mx.gluon.loss.L2Loss()
    # end method


    def train_op(self, X_in):
        X_in = self.from_numpy(X_in)
        G_in = mx.nd.random_normal(shape=(X_in.shape[0], self.G_size))

        ones = mx.nd.ones((X_in.shape[0]))
        zeros = mx.nd.zeros((X_in.shape[0]))

        with mx.gluon.autograd.record(train_mode=True):
            G_prob = self.d(self.g(G_in))
            X_prob = self.d(X_in)
            D_loss = self.loss(X_prob, ones) + self.loss(G_prob, zeros)
        D_loss.backward()
        self.d_optim.step(X_in.shape[0])
        
        for _ in range(2):
            with mx.gluon.autograd.record(train_mode=True):
                G_out = self.g(G_in)
                G_loss = self.loss(self.d(G_out), ones)
            G_loss.backward()
            self.g_optim.step(X_in.shape[0])

        mse_loss = self.mse(G_out, X_in)

        return (mx.nd.mean(G_loss).asscalar(),
                mx.nd.mean(D_loss).asscalar(),
                mx.nd.argmax(mx.nd.softmax(X_prob), 1).asnumpy(),
                mx.nd.argmax(mx.nd.softmax(G_prob), 1).asnumpy(),
                mx.nd.mean(mse_loss).asscalar())
    # end method


    def from_numpy(self, *args):
        data = []
        for _arr in args:
            arr = mx.nd.zeros(_arr.shape)
            arr[:] = _arr
            data.append(arr)
        return data if len(data) > 1 else data[0]
    # end method
# end class
