from __future__ import print_function
import torch
import sklearn
import numpy as np
import math


class Autoencoder(torch.nn.Module):
    def __init__(self, n_in, encoder_units, noise_level=0.5, drop_proba=0.2):
        super(Autoencoder, self).__init__()
        self.n_in = n_in
        self.encoder_units = encoder_units
        self.decoder_units = list(reversed(encoder_units))
        self.noise_level = noise_level
        self.drop_proba = drop_proba
        self.build_model()
    # end constructor


    def build_model(self):
        self.dropout = torch.nn.Dropout(self.drop_proba)
        self.encoder = torch.nn.Sequential(*self._encoder_dense())
        self.decoder = torch.nn.Sequential(*self._decoder_dense())
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters())
    # end method


    def _encoder_dense(self):
        dense = []
        forward = [self.n_in] + self.encoder_units
        for i in range(1, len(forward)-1):
            dense.append(torch.nn.Linear(forward[i-1], forward[i]))
            dense.append(torch.nn.ReLU())
        dense.append(torch.nn.Linear(forward[-2], forward[-1]))
        return dense
    # end method


    def _decoder_dense(self):
        dense = []
        forward = self.decoder_units
        for i in range(1, len(forward)):
            dense.append(torch.nn.Linear(forward[i-1], forward[i]))
            dense.append(torch.nn.ReLU())
        dense.append(torch.nn.Linear(forward[-1], self.n_in))
        dense.append(torch.nn.Sigmoid())
        return dense
    # end method

    
    def forward(self, inputs):
        noisy = self.dropout(inputs) + self.noise_level * torch.autograd.Variable(
                                       torch.randn(inputs.size(0), inputs.size(1)))
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return encoded, decoded
    # end method
    

    def fit(self, X, n_epoch=10, batch_size=128, en_shuffle=True):
        for epoch in range(n_epoch):
            if en_shuffle:
                print("Data Shuffled")
                X = sklearn.utils.shuffle(X)
            for local_step, X_batch in enumerate(self.gen_batch(X, batch_size)):
                inputs = torch.autograd.Variable(torch.from_numpy(X_batch.astype(np.float32)))
                _, outputs = self.forward(inputs)
                
                loss = self.criterion(outputs, inputs)
                self.optimizer.zero_grad()                             # clear gradients for this training step
                loss.backward()                                        # backpropagation, compute gradients
                self.optimizer.step()                                  # apply gradients
                if local_step % 50 == 0:
                    print ("Epoch %d/%d | Step %d/%d | BCE loss: %.4f |"
                           %(epoch+1, n_epoch, local_step, len(X)//batch_size, loss.data[0]))
    # end method

    
    def transform(self, X, batch_size=128):
        self.eval()
        res = []
        for X_batch in self.gen_batch(X, batch_size):
            inputs = torch.autograd.Variable(torch.from_numpy(X_batch.astype(np.float32)))
            res.append(self.forward(inputs)[0].data.numpy())
        return np.vstack(res)
    # end method predict


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method
# end class Autoencoder
