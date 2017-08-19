from __future__ import print_function
import torch
import numpy as np
import math
from sklearn.utils import shuffle


class ConvLSTMClassifier(torch.nn.Module):
    def __init__(self, vocab_size, n_out=2, embedding_dim=128, n_filters=64, kernel_size=5, pool_size=4,
                 cell_size=70, dropout=0.2, stateful=False, grad_clip=5.0):
        super(ConvLSTMClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.cell_size = cell_size
        self.dropout = dropout
        self.stateful = stateful
        self.n_out = n_out
        self.grad_clip = grad_clip
        self.build_model()
    # end constructor


    def build_model(self):
        self.encoder = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        self.conv1d = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels = self.embedding_dim,
                            out_channels = self.n_filters,
                            kernel_size = self.kernel_size),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size = self.pool_size))
        self.lstm = torch.nn.LSTM(input_size = self.n_filters,
                                  hidden_size = self.cell_size,
                                  dropout = self.dropout,
                                  batch_first = True)
        self.fc = torch.nn.Linear(self.cell_size, self.n_out)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters())
    # end method build_model    


    def forward(self, X, X_lens, init_state=None, is_training=True):
        embedded = self.encoder(X)
        conv_out = self.conv1d(embedded.permute(0, 2, 1))
        
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            conv_out.permute(0, 2, 1), X_lens, batch_first=True)
        lstm_out, final_state = self.lstm(packed, init_state)
        h_n, c_n = final_state

        logits = self.fc(torch.squeeze(h_n, 0))
        return logits, final_state
    # end method forward


    def fit(self, X, y, n_epoch=10, batch_size=32, en_shuffle=False):
        global_step = 0
        n_batch = int(len(X) / batch_size)
        total_steps = int(n_epoch * n_batch)

        X, y, X_lens = self.sort_pad(X, y)
        X_lens = [l // self.pool_size for l in X_lens]

        for epoch in range(n_epoch):
            state = None
            if en_shuffle:
                X, y = shuffle(X, y)
                print("Data Shuffled")
            for local_step, (X_batch, y_batch, X_lens_batch) in enumerate(zip(self.gen_batch(X, batch_size),
                                                                              self.gen_batch(y, batch_size),
                                                                              self.gen_batch(X_lens, batch_size))):
                inputs = torch.autograd.Variable(torch.from_numpy(X_batch.astype(np.int64)))
                labels = torch.autograd.Variable(torch.from_numpy(y_batch.astype(np.int64)))
                
                if (self.stateful) and (len(X_batch) == batch_size):
                    preds, state = self.forward(inputs, X_lens_batch, state)
                    state = (torch.autograd.Variable(state[0].data), torch.autograd.Variable(state[1].data))
                else:
                    preds, _ = self.forward(inputs, X_lens_batch)

                loss = self.criterion(preds, labels)                   # cross entropy loss
                self.optimizer, lr = self.adjust_lr(self.optimizer, global_step, total_steps)
                self.optimizer.zero_grad()                             # clear gradients for this training step
                loss.backward()                                        # backpropagation, compute gradients
                torch.nn.utils.clip_grad_norm(self.parameters(), self.grad_clip)
                self.optimizer.step()                                  # apply gradients
                global_step += 1

                preds = torch.max(preds,1)[1].data.numpy().squeeze()
                acc = (preds == y_batch).mean()
                if local_step % 50 == 0:
                    print ('Epoch [%d/%d] | Step [%d/%d] | Loss: %.4f | Acc: %.4f | LR: %.4f'
                           %(epoch+1, n_epoch, local_step, n_batch, loss.data[0], acc, lr))
    # end method fit


    def evaluate(self, X_test, y_test, batch_size=32):
        self.eval()

        correct = 0
        total = 0
        state = None

        X_test, y_test, X_lens = self.sort_pad(X_test, y_test)
        X_lens = [l // self.pool_size for l in X_lens]

        for X_batch, y_batch, X_lens_batch in zip(self.gen_batch(X_test, batch_size),
                                                  self.gen_batch(y_test, batch_size),
                                                  self.gen_batch(X_lens, batch_size)):
            inputs = torch.autograd.Variable(torch.from_numpy(X_batch.astype(np.int64)))
            labels = torch.from_numpy(y_batch.astype(np.int64))

            if (self.stateful) and (len(X_batch) == batch_size):
                preds, state = self.forward(inputs, X_lens_batch, state, is_training=False)
                state = (torch.autograd.Variable(state[0].data), torch.autograd.Variable(state[1].data))
            else:
                preds, _ = self.forward(inputs, X_lens_batch, is_training=False)

            _, preds = torch.max(preds.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum()
        print('Test Accuracy of the model: %.4f' % (float(correct) / total)) 
    # end method evaluate


    def sort_pad(self, X, y, thres=250):
        X = np.array(X)
        lens = [len(x) for x in X]
        max_len = max(lens)
        if max_len >= thres:
            max_len = thres
        idx = list(reversed(np.argsort(lens)))

        X = X[idx].tolist()
        new_lens = []
        for i, x in enumerate(X):
            if len(x) >= max_len:
                X[i] = x[:max_len]
                new_lens.append(max_len)
            else:
                X[i] = x + [0] * (max_len - len(x))
                new_lens.append(len(x))
        X = np.array(X)

        y = np.array(y[idx])
        print("Sorting and Padding", X.shape, y.shape)
        return X, y, new_lens
    # end method sort_pad


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i + batch_size]
    # end method gen_batch


    def adjust_lr(self, optimizer, current_step, total_steps):
        max_lr = 0.005
        min_lr = 0.001
        decay_rate = math.log(min_lr/max_lr) / (-total_steps)
        lr = max_lr * math.exp(-decay_rate * current_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer, lr
    # end method adjust_lr
# end class RNNClassifier
