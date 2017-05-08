import torch 
from torch import nn
from torch.autograd import Variable
import numpy as np


class RNNClassifier(nn.Module):
    def __init__(self, n_in, cell_size, n_layer, n_out):
        super(RNNClassifier, self).__init__()
        self.n_in = n_in
        self.cell_size = cell_size
        self.n_layer = n_layer
        self.n_out = n_out
        self.build_model()
    # end constructor


    def build_model(self):
        self.lstm = nn.LSTM(self.n_in, self.cell_size, self.n_layer, batch_first=True)
        self.fc = nn.Linear(self.cell_size, self.n_out)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    # end method build_model    


    def forward(self, X):
        h_0 = Variable(torch.zeros(self.n_layer, X.size(0), self.cell_size)) # set initial states  
        c_0 = Variable(torch.zeros(self.n_layer, X.size(0), self.cell_size)) # set initial states 
        out, (h_n, c_n) = self.lstm(X, (h_0, c_0))                                # forward propagate
        out = self.fc(out[:, -1, :])                                              # decode hidden state of last time step
        return out
    # end method forward


    def fit(self, X, y, num_epochs, batch_size):
        for epoch in range(num_epochs):
            i = 0
            for X_train_batch, y_train_batch in zip(self.gen_batch(X, batch_size),
                                                    self.gen_batch(y, batch_size)):
                images = Variable(torch.from_numpy(X_train_batch.astype(np.float32)))
                labels = Variable(torch.from_numpy(y_train_batch.astype(np.int64)))
                
                pred = self.forward(images)             # rnn output
                loss = self.criterion(pred, labels)     # cross entropy loss
                self.optimizer.zero_grad()              # clear gradients for this training step
                loss.backward()                         # backpropagation, compute gradients
                self.optimizer.step()                   # apply gradients
                i+=1 
                acc = np.equal(torch.max(pred,1)[1].data.numpy().squeeze(), y_train_batch).astype(float).mean()
                if (i+1) % 100 == 0:
                    print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Acc: %.4f'
                           %(epoch+1, num_epochs, i+1, int(len(X)/batch_size), loss.data[0], acc))
    # end method fit


    def evaluate(self, X_test, y_test, batch_size):
        correct = 0
        total = 0
        for X_test_batch, y_test_batch in zip(self.gen_batch(X_test, batch_size),
                                              self.gen_batch(y_test, batch_size)):
            images = Variable(torch.from_numpy(X_test_batch.astype(np.float32)))
            labels = torch.from_numpy(y_test_batch.astype(np.int64))
            outputs = self.forward(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total)) 
    # end method evaluate


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i + batch_size]
    # end method gen_batch
# end class RNNClassifier
