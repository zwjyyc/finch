import torch
import numpy as np


class Generator(torch.nn.Module):
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
        self.linear = torch.nn.Linear(self.G_size, np.prod(self.shape_trace[0]))
        self.deconv = torch.nn.Sequential(*self._net())
    # end method


    def _net(self):
        net = []
        for i in range(1, len(self.shape_trace)):
            net.append(torch.nn.ConvTranspose2d(
                self.shape_trace[i-1][2], self.shape_trace[i][2], self.kernel_size, self.stride, self.padding))
            net.append(torch.nn.BatchNorm2d(self.shape_trace[i][2], momentum=0.9))
            net.append(torch.nn.LeakyReLU(0.2))
        net.append(torch.nn.ConvTranspose2d(
            self.shape_trace[-1][2], self.img_ch, self.kernel_size, self.stride, self.padding))
        return net
    # end method


    def forward(self, X):
        X = torch.nn.functional.leaky_relu(self.linear(X), 0.2)
        X = X.view(-1, self.shape_trace[0][2], self.shape_trace[0][0], self.shape_trace[0][1])
        X = self.deconv(X)
        return torch.nn.functional.tanh(X)
    # end method
# end class


class Discriminator(torch.nn.Module):
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
        self.conv = torch.nn.Sequential(*self._net())
        self.linear = torch.nn.Linear(np.prod(self.shape_trace[0]), 1)
    # end method


    def _net(self):
        shape_trace = [(self.img_size[0], self.img_size[1], self.img_ch)] + list(reversed(self.shape_trace))
        net = []
        for i in range(1, len(shape_trace)):
            net.append(torch.nn.Conv2d(
                shape_trace[i-1][2], shape_trace[i][2], self.kernel_size, self.stride, self.padding))
            net.append(torch.nn.BatchNorm2d(shape_trace[i][2], momentum=0.9))
            net.append(torch.nn.LeakyReLU(0.2))
        return net
    # end method


    def forward(self, X):
        X = self.conv(X)
        X = X.view(-1, np.prod(self.shape_trace[0]))
        X = self.linear(X)
        return torch.nn.functional.sigmoid(X)
    # end method
# end class


class GAN:
    def __init__(self, G_size, img_size, img_ch, shape_trace, kernel_size=4, stride=2, padding=1):
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
        self.g = Generator(self.G_size, self.img_ch, self.shape_trace,
                           self.kernel_size, self.stride, self.padding)
        self.d = Discriminator(self.img_size, self.img_ch, self.shape_trace,
                               self.kernel_size, self.stride, self.padding)
        self.bce_loss = torch.nn.BCELoss()
        self.mse_loss = torch.nn.MSELoss()
        self.g_optim = torch.optim.Adam(self.g.parameters(), 2e-4, (0.5, 0.999))
        self.d_optim = torch.optim.Adam(self.d.parameters(), 2e-4, (0.5, 0.999))
    # end method


    def train_op(self, X_in):
        X_in = torch.autograd.Variable(torch.from_numpy(X_in.astype(np.float32)))
        G_in = torch.autograd.Variable(torch.randn(X_in.size(0), self.G_size))

        G_out = self.g(G_in)
        G_prob = self.d(G_out)
        X_prob = self.d(X_in)
        
        ones = torch.autograd.Variable(torch.ones(G_prob.size(0), G_prob.size(1)) - 0.1)
        zeros = torch.autograd.Variable(torch.zeros(G_prob.size(0), G_prob.size(1)) + 0.1)

        D_loss = self.bce_loss(X_prob, ones) + self.bce_loss(G_prob, zeros)
        self.d_optim.zero_grad()
        D_loss.backward()
        self.d_optim.step()
        
        for _ in range(2):
            G_loss = self.bce_loss(self.d(self.g(G_in)), ones)
            self.g_optim.zero_grad()
            G_loss.backward()
            self.g_optim.step()

        mse_loss = self.mse_loss(G_out, X_in)

        return G_loss, D_loss, X_prob, G_prob, mse_loss
    # end method
# end class
