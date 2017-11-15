from __future__ import division
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class RNNClassifier(torch.nn.Module):
    def __init__(self, n_in, n_out, cell_size=128, n_layer=2, stateful=False, dropout=0.2):
        super(RNNClassifier, self).__init__()
        self.n_in = n_in
        self.cell_size = cell_size
        self.n_layer = n_layer
        self.n_out = n_out
        self.stateful = stateful
        self.dropout = dropout
        self.build_model()
    # end constructor


    def build_model(self):
        self.lstm = torch.nn.LSTM(self.n_in, self.cell_size, self.n_layer,
                                  batch_first=True, dropout=self.dropout)
        self.fc = torch.nn.Linear(self.cell_size, self.n_out)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    # end method build_model    


    def forward(self, X, init_state=None):
        rnn_out, final_state = self.lstm(X, init_state) # forward propagate
        last_time_step = self.fc(rnn_out[:, -1, :])     # decode hidden state of last time step
        return last_time_step, final_state
    # end method forward


    def fit(self, X, y, num_epochs, batch_size):
        dataset = TensorDataset(data_tensor = torch.from_numpy(X.astype(np.float32)),
                                target_tensor = torch.from_numpy(y.astype(np.int64)))
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(num_epochs):
            state = None
            for i, (X_batch, y_batch) in enumerate(loader):
                inputs = torch.autograd.Variable(X_batch)
                labels = torch.autograd.Variable(y_batch)
                
                if (self.stateful) and (len(X_batch) == batch_size):
                    preds, state = self.forward(inputs, state)
                    state = (torch.autograd.Variable(state[0].data),
                             torch.autograd.Variable(state[1].data))
                else:
                    preds, _ = self.forward(inputs)

                loss = self.criterion(preds, labels)     # cross entropy loss
                self.optimizer.zero_grad()               # clear gradients for this training step
                loss.backward()                          # backpropagation, compute gradients
                self.optimizer.step()                    # apply gradients

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
        state = None
        dataset = TensorDataset(data_tensor = torch.from_numpy(X_test.astype(np.float32)),
                                target_tensor = torch.from_numpy(y_test.astype(np.int64)))
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        for X_batch, y_batch in loader:
            inputs = torch.autograd.Variable(X_batch)
            labels = y_batch

            if (self.stateful) and (len(X_batch) == batch_size):
                preds, state = self.forward(inputs, state)
                state = (torch.autograd.Variable(state[0].data),
                         torch.autograd.Variable(state[1].data))
            else:
                preds, _ = self.forward(inputs)

            _, preds = torch.max(preds.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum()
        print('Test Accuracy of the model on the 10000 test images: %.4f %%' % (correct / total)) 
    # end method evaluate


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i + batch_size]
    # end method gen_batch
# end class RNNClassifier
