from __future__ import division
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class MLPClassifier(torch.nn.Module):
    def __init__(self, n_in, hidden_units, n_out):
        super(MLPClassifier, self).__init__()
        self.n_in = n_in
        self.hidden_units = hidden_units
        self.n_out = n_out
        self.build_model()
    # end constructor


    def build_model(self):            
        self.mlp = torch.nn.Sequential(*self._dense())
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    # end method build_model    


    def _dense(self):
        dense = []
        forward = [self.n_in] + self.hidden_units
        for i in range(len(forward)-1):
            dense.append(torch.nn.Linear(forward[i], forward[i+1]))
            dense.append(torch.nn.BatchNorm2d(forward[i+1]))
            dense.append(torch.nn.ReLU())
        dense.append(torch.nn.Linear(self.hidden_units[-1], self.n_out))
        return dense
    # end method _dense         
    

    def forward(self, X):
        Y = self.mlp(X)
        return Y
    # end method forward


    def fit(self, X, y, num_epochs, batch_size):
        dataset = TensorDataset(data_tensor = torch.from_numpy(X.astype(np.float32)),
                                target_tensor = torch.from_numpy(y.astype(np.int64)))
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(num_epochs):
            for i, (X_batch, y_batch) in enumerate(loader):
                inputs = torch.autograd.Variable(X_batch)
                labels = torch.autograd.Variable(y_batch)

                preds = self.forward(inputs)            # cnn output
                loss = self.criterion(preds, labels)    # cross entropy loss
                self.optimizer.zero_grad()              # clear gradients for this training step
                loss.backward()                         # backpropagation, compute gradients
                self.optimizer.step()                   # apply gradients
                preds = torch.max(preds, 1)[1].data.numpy().squeeze()
                acc = (preds == y_batch.numpy()).mean()
                if (i+1) % 100 == 0:
                    print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Acc: %.4f'
                           %(epoch+1, num_epochs, i+1, int(len(X)/batch_size), loss.data[0], acc))
    # end method fit


    def evaluate(self, X_test, y_test, batch_size):
        self.eval()
        correct = 0
        total = 0
        dataset = TensorDataset(data_tensor = torch.from_numpy(X_test.astype(np.float32)),
                                target_tensor = torch.from_numpy(y_test.astype(np.int64)))
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        for X_batch, y_batch in loader:
            inputs = torch.autograd.Variable(X_batch)
            labels = y_batch

            preds = self.forward(inputs)
            _, preds = torch.max(preds.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum()
        print('Test Accuracy of the model on the 10000 test images: %.4f' % (correct / total)) 
    # end method evaluate


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i + batch_size]
    # end method gen_batch
# end class RNNClassifier
