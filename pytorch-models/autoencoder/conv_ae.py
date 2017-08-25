from __future__ import print_function
import torch
import sklearn
import numpy as np
import math


class Autoencoder(torch.nn.Module):
    def __init__(self, img_size, img_ch, kernel_size=(5,5)):
        super(Autoencoder, self).__init__()
        self.img_size = img_size
        self.img_ch = img_ch
        self.kernel_size = kernel_size
        self.build_model()
    # end constructor


    def build_model(self):
        self.encoder = torch.nn.Conv2d(self.img_ch, 32, kernel_size=self.kernel_size)
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters())
    # end method

    
    def forward(self, inputs):
        encoded = self.encoder(inputs)
        encoded = torch.nn.functional.relu(encoded)
        decoded = torch.nn.functional.conv_transpose2d(encoded, self.encoder.weight)
        decoded = torch.nn.functional.sigmoid(decoded)
        return decoded
    # end method
    

    def fit(self, X, n_epoch=5, batch_size=128, en_shuffle=True):
        for epoch in range(n_epoch):
            if en_shuffle:
                X = sklearn.utils.shuffle(X)
                print("Data Shuffled")
            for local_step, X_batch in enumerate(self.gen_batch(X, batch_size)):
                inputs = torch.autograd.Variable(torch.from_numpy(X_batch.astype(np.float32)))
                outputs = self.forward(inputs)
                
                loss = self.criterion(outputs, inputs)
                self.optimizer.zero_grad()                             # clear gradients for this training step
                loss.backward()                                        # backpropagation, compute gradients
                self.optimizer.step()                                  # apply gradients
                if local_step % 50 == 0:
                    print ("Epoch %d/%d | Step %d/%d | BCE loss: %.4f |"
                           %(epoch+1, n_epoch, local_step, len(X)//batch_size, loss.data[0]))
    # end method


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method
# end class Autoencoder
