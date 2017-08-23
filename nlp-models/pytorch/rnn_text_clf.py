from __future__ import print_function
import torch
import numpy as np
import math
from sklearn.utils import shuffle


class RNNTextClassifier(torch.nn.Module):
    def __init__(self, vocab_size, n_out=2, embedding_dim=128, cell_size=128, n_layer=1, stateful=False,
                 dropout=0.2, grad_clip=5.0):
        super(RNNTextClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.cell_size = cell_size
        self.n_layer = n_layer
        self.n_out = n_out
        self.stateful = stateful
        self.dropout = dropout
        self.grad_clip = grad_clip
        self.build_model()
    # end constructor


    def build_model(self):
        self.encoder = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = torch.nn.LSTM(self.embedding_dim, self.cell_size, self.n_layer,
                                  batch_first=True, dropout=self.dropout)
        self.fc = torch.nn.Linear(self.cell_size, self.n_out)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters())
    # end method build_model    


    def forward(self, X, X_lens, init_state=None):
        embedded = self.encoder(X)

        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, X_lens, batch_first=True)
        rnn_out, final_state = self.lstm(packed, init_state)
        # rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        h_n, c_n = final_state

        logits = self.fc(torch.squeeze(h_n, 0))
        return logits, final_state
    # end method forward


    def fit(self, X, y, n_epoch=10, batch_size=32, en_shuffle=False):
        X, y, X_lens = self.sort_pad(X, y)
        global_step = 0
        n_batch = int(len(X) / batch_size)
        total_steps = int(n_epoch * n_batch)

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
        for X_batch, y_batch, X_lens_batch in zip(self.gen_batch(X_test, batch_size),
                                                  self.gen_batch(y_test, batch_size),
                                                  self.gen_batch(X_lens, batch_size)):
            inputs = torch.autograd.Variable(torch.from_numpy(X_batch.astype(np.int64)))
            labels = torch.from_numpy(y_batch.astype(np.int64))

            if (self.stateful) and (len(X_batch) == batch_size):
                preds, state = self.forward(inputs, X_lens_batch, state)
                state = (torch.autograd.Variable(state[0].data), torch.autograd.Variable(state[1].data))
            else:
                preds, _ = self.forward(inputs, X_lens_batch)

            _, preds = torch.max(preds.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum()
        print('Test Accuracy of the model: %.4f' % (float(correct) / total)) 
    # end method evaluate


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


    def sort_pad(self, X, y, thres=250):
        lens = [len(x) for x in X]
        max_len = max(lens)
        if max_len >= thres:
            max_len = thres
        idx = list(reversed(np.argsort(lens)))

        X = np.array(X)[idx].tolist()
        new_lens = []
        for i, x in enumerate(X):
            if len(x) >= max_len:
                X[i] = x[:max_len]
                new_lens.append(max_len)
            else:
                X[i] = x + [0] * (max_len - len(x))
                new_lens.append(len(x))
        X = np.array(X)

        y = np.array(y)[idx]
        print("Sorting and Padding", X.shape, y.shape)
        return X, y, new_lens
# end class RNNClassifier
