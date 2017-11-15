from __future__ import print_function
import torch
import sklearn
import numpy as np


class Autoencoder(torch.nn.Module):
    def __init__(self, n_in, encoder_units):
        super(Autoencoder, self).__init__()
        self.n_in = n_in
        self.encoder_units = encoder_units
        self.decoder_units = list(reversed(encoder_units))
        self.build_model()
    # end constructor


    def build_model(self):
        self.encoder = torch.nn.Sequential(*self._encoder_dense())
        self.mean = torch.nn.Linear(self.encoder_units[-2], self.encoder_units[-1])
        self.gamma = torch.nn.Linear(self.encoder_units[-2], self.encoder_units[-1])
        self.decoder = torch.nn.Sequential(*self._decoder_dense())
        self.bce_loss = torch.nn.BCELoss(size_average=False)
        self.optimizer = torch.optim.Adam(self.parameters())
    # end build_model


    def _encoder_dense(self):
        dense = []
        forward = [self.n_in] + self.encoder_units
        for i in range(1, len(forward)-1):
            dense.append(torch.nn.Linear(forward[i-1], forward[i]))
            dense.append(torch.nn.ELU())
        return dense
    # end method


    def _decoder_dense(self):
        dense = []
        forward = self.decoder_units
        for i in range(1, len(forward)):
            dense.append(torch.nn.Linear(forward[i-1], forward[i]))
            dense.append(torch.nn.ELU())
        dense.append(torch.nn.Linear(forward[-1], self.n_in))
        dense.append(torch.nn.Sigmoid())
        return dense
    # end method


    def forward(self, inputs):
        encoded = self.encoder(inputs)

        mean = self.mean(encoded)
        gamma = self.gamma(encoded)
        kl_loss = 0.5 * torch.sum(torch.exp(gamma) + mean**2 - 1. - gamma)

        noise = torch.autograd.Variable(torch.randn(gamma.size(0), gamma.size(1)))
        encoded = mean + torch.exp(0.5 * gamma) * noise

        decoded = self.decoder(encoded)
        return decoded, kl_loss
    # end method


    def fit(self, X, n_epoch=10, batch_size=128, en_shuffle=True):
        for epoch in range(n_epoch):
            if en_shuffle:
                print("Data Shuffled")
                X = sklearn.utils.shuffle(X)
            for local_step, X_batch in enumerate(self.gen_batch(X, batch_size)):
                inputs = torch.autograd.Variable(torch.from_numpy(X_batch.astype(np.float32)))
                outputs, kl_loss = self.forward(inputs)

                bce_loss = self.bce_loss(outputs, inputs)
                loss = bce_loss + kl_loss                   
                self.optimizer.zero_grad()                             # clear gradients for this training step
                loss.backward()                                        # backpropagation, compute gradients
                self.optimizer.step()                                  # apply gradients
                if local_step % 50 == 0:
                    print ("Epoch %d/%d | Step %d/%d | train loss: %.4f | bce loss: %.4f | kl loss: %.4f"
                           %(epoch+1, n_epoch, local_step, len(X)//batch_size, loss.data[0], bce_loss.data[0], kl_loss.data[0]))
    # end method


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method
# end class Autoencoder
