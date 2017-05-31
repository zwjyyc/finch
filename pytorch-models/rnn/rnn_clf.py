import torch
import numpy as np


class RNNClassifier(torch.nn.Module):
    def __init__(self, n_in, n_out, cell_size=128, n_layer=2, stateful=False):
        super(RNNClassifier, self).__init__()
        self.n_in = n_in
        self.cell_size = cell_size
        self.n_layer = n_layer
        self.n_out = n_out
        self.stateful = stateful
        self.build_model()
    # end constructor


    def build_model(self):
        self.lstm = torch.nn.LSTM(self.n_in, self.cell_size, self.n_layer, batch_first=True)
        self.fc = torch.nn.Linear(self.cell_size, self.n_out)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    # end method build_model    


    def forward(self, X, init_state=None):
        Y, final_state = self.lstm(X, init_state) # forward propagate
        Y = self.fc(Y[:, -1, :])                  # decode hidden state of last time step
        return Y, final_state
    # end method forward


    def fit(self, X, y, num_epochs, batch_size):
        for epoch in range(num_epochs):
            i = 0
            state = None
            for X_batch, y_batch in zip(self.gen_batch(X, batch_size),
                                                    self.gen_batch(y, batch_size)):
                X_train_batch = torch.autograd.Variable(torch.from_numpy(X_batch.astype(np.float32)))
                y_train_batch = torch.autograd.Variable(torch.from_numpy(y_batch.astype(np.int64)))
                
                if (self.stateful) and (len(X_batch) == batch_size):
                    y_pred_batch, state = self.forward(X_train_batch, state)
                    state = (torch.autograd.Variable(state[0].data), torch.autograd.Variable(state[1].data))
                else:
                    y_pred_batch, _ = self.forward(X_train_batch)

                loss = self.criterion(y_pred_batch, y_train_batch)     # cross entropy loss
                self.optimizer.zero_grad()                             # clear gradients for this training step
                loss.backward()                                        # backpropagation, compute gradients
                self.optimizer.step()                                  # apply gradients
                i+=1 
                acc = (torch.max(y_pred_batch,1)[1].data.numpy().squeeze() == y_batch).astype(float).mean()
                if (i+1) % 100 == 0:
                    print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Acc: %.4f'
                           %(epoch+1, num_epochs, i+1, int(len(X)/batch_size), loss.data[0], acc))
    # end method fit


    def evaluate(self, X_test, y_test, batch_size):
        correct = 0
        total = 0
        state = None
        for X_batch, y_batch in zip(self.gen_batch(X_test, batch_size),
                                              self.gen_batch(y_test, batch_size)):
            X_test_batch = torch.autograd.Variable(torch.from_numpy(X_batch.astype(np.float32)))
            y_test_batch = torch.from_numpy(y_batch.astype(np.int64))

            if (self.stateful) and (len(X_batch) == batch_size):
                y_pred_batch, state = self.forward(X_test_batch, state)
                state = (torch.autograd.Variable(state[0].data), torch.autograd.Variable(state[1].data))
            else:
                y_pred_batch, _ = self.forward(X_test_batch)

            _, y_pred_batch = torch.max(y_pred_batch.data, 1)
            total += y_test_batch.size(0)
            correct += (y_pred_batch == y_test_batch).sum()
        print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total)) 
    # end method evaluate


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i + batch_size]
    # end method gen_batch
# end class RNNClassifier
