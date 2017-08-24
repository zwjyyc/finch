import torch
import sklearn
import numpy as np
import math


class Autoencoder(torch.nn.Module):
    def __init__(self, n_in, n_hidden=1000, sparsity_target=0.1, sparsity_weight=0.2, lr=0.01):
        super(Autoencoder, self).__init__()
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight
        self.lr = lr
        self.build_model()
    # end constructor


    def build_model(self):
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.n_in, self.n_hidden),
            torch.nn.Sigmoid())
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.n_hidden, self.n_in),
            torch.nn.Sigmoid())
        self.bce_loss = torch.nn.BCELoss(size_average=False)
        self.optimizer = torch.optim.Adam(self.parameters(), self.lr)
    # end method


    def forward(self, inputs):
        hidden = self.encoder(inputs)
        hidden_mean = torch.mean(hidden, dim=0)
        sparsity_loss = torch.sum(self.kl_divergence(self.sparsity_target, hidden_mean))
        return self.decoder(hidden), sparsity_loss
    # end method        


    def kl_divergence(self, p, q):
        return p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q)) # Kullback Leibler divergence
    # end method


    def fit(self, X, n_epoch=10, batch_size=128, en_shuffle=True):
        for epoch in range(n_epoch):
            if en_shuffle:
                print("Data Shuffled")
                X = sklearn.utils.shuffle(X)
            for local_step, X_batch in enumerate(self.gen_batch(X, batch_size)):
                inputs = torch.autograd.Variable(torch.from_numpy(X_batch.astype(np.float32)))
                outputs, sparsity_loss = self.forward(inputs)

                bce_loss = self.bce_loss(outputs, inputs)
                loss = bce_loss + self.sparsity_weight * sparsity_loss                   
                self.optimizer.zero_grad()                             # clear gradients for this training step
                loss.backward()                                        # backpropagation, compute gradients
                self.optimizer.step()                                  # apply gradients
                if local_step % 50 == 0:
                    print ("Epoch %d/%d | Step %d/%d | train loss: %.4f | bce loss: %.4f | sparsity loss: %.4f"
                           %(epoch+1, n_epoch, local_step, len(X)//batch_size,
                             loss.data[0], bce_loss.data[0], sparsity_loss.data[0]))
    # end method


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method
# end class Autoencoder
