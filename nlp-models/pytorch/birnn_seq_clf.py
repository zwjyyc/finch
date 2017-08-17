import torch
import numpy as np
import math


class BiRNN(torch.nn.Module):
    def __init__(self, vocab_size, n_out, embedding_dim=128, cell_size=128, dropout=0.2):
        super(BiRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.cell_size = cell_size
        self.n_out = n_out
        self.dropout = dropout
        self.build_model()
    # end constructor


    def build_model(self):
        self.encoder = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        self.birnn = torch.nn.LSTM(input_size=self.embedding_dim,
                                   hidden_size=self.cell_size,
                                   batch_first=True,
                                   dropout=self.dropout,
                                   bidirectional=True)
        self.fc = torch.nn.Linear(2 * self.cell_size, self.n_out)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters())
    # end method build_model    


    def forward(self, X):
        birnn_out, _ = self.birnn(self.encoder(X), None)
        reshaped = birnn_out.contiguous().view(-1, 2*self.cell_size)
        logits = self.fc(reshaped)                  
        return logits
    # end method forward


    def fit(self, X, Y, n_epoch=10, batch_size=128, en_shuffle=True):
        global_step = 0
        n_batch = len(X) / batch_size
        total_steps = int(n_epoch * n_batch)
        for epoch in range(n_epoch):
            if en_shuffle:
                shuffled = np.random.permutation(len(X))
                X = X[shuffled]
                Y = Y[shuffled]
            for local_step, (X_batch, Y_batch) in enumerate(zip(self.gen_batch(X, batch_size),
                                                                self.gen_batch(Y, batch_size))):
                y_batch = Y_batch.ravel()
                inputs = torch.autograd.Variable(torch.from_numpy(X_batch.astype(np.int64)))
                labels = torch.autograd.Variable(torch.from_numpy(y_batch.astype(np.int64)))

                preds = self.forward(inputs)
                loss = self.criterion(preds, labels)     # cross entropy loss
                self.optimizer, lr = self.adjust_lr(self.optimizer, global_step, total_steps)
                self.optimizer.zero_grad()                             # clear gradients for this training step
                loss.backward()                                        # backpropagation, compute gradients
                self.optimizer.step()                                  # apply gradients

                global_step += 1
                preds = torch.max(preds, 1)[1].data.numpy().squeeze()
                acc = (preds == y_batch).mean()
                if local_step % 100 == 0:
                    print ('Epoch [%d/%d] | Step [%d/%d] | Loss: %.4f | Acc: %.4f | LR: %.4f'
                           %(epoch+1, n_epoch, local_step, n_batch, loss.data[0], acc, lr))
    # end method fit


    def evaluate(self, X_test, Y_test, batch_size=128):
        self.birnn.eval()

        correct = 0
        total = 0
        for X_batch, Y_batch in zip(self.gen_batch(X_test, batch_size), self.gen_batch(Y_test, batch_size)):
            y_batch = Y_batch.ravel()
            inputs = torch.autograd.Variable(torch.from_numpy(X_batch.astype(np.int64)))
            labels = torch.from_numpy(y_batch.astype(np.int64))
            preds = self.forward(inputs)

            _, preds = torch.max(preds.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum()
        print('Test Accuracy of the model: %.4f' % (float(correct) / total)) 
    # end method evaluate

    
    def infer(self, x):
        x = np.atleast_2d(x)
        x = torch.autograd.Variable(torch.from_numpy(x.astype(np.int64)))
        y = self.forward(x)
        return y.data.numpy()
    # end method infer


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i + batch_size]
    # end method gen_batch


    def adjust_lr(self, optimizer, current_step, total_steps):
        max_lr = 0.005
        min_lr = 0.0005
        decay_rate = math.log(min_lr/max_lr) / (-total_steps)
        lr = max_lr * math.exp(-decay_rate * current_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer, lr
    # end method adjust_lr
# end class RNNClassifier
