import torch
import numpy as np
import math
import tensorflow as tf
from sklearn.utils import shuffle


class RNNTextClassifier(torch.nn.Module):
    def __init__(self, vocab_size, n_out=2, cell_size=128, n_layer=1, stateful=False):
        super(RNNTextClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.cell_size = cell_size
        self.n_layer = n_layer
        self.n_out = n_out
        self.stateful = stateful
        self.build_model()
    # end constructor


    def build_model(self):
        self.encoder = torch.nn.Embedding(self.vocab_size, self.cell_size)
        self.lstm = torch.nn.LSTM(self.cell_size, self.cell_size, self.n_layer, batch_first=True)
        self.fc = torch.nn.Linear(self.cell_size, self.n_out)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters())
    # end method build_model    


    def forward(self, X, init_state=None):
        X = self.encoder(X)
        Y, final_state = self.lstm(X, init_state) # forward propagate
        Y = self.fc(Y[:, -1, :])                  # decode hidden state of last time step
        return Y, final_state
    # end method forward


    def fit(self, X, y, n_epoch=10, batch_size=32):
        global_step = 0
        n_batch = len(X) / batch_size
        total_steps = int(n_epoch * n_batch)
        for epoch in range(n_epoch):
            X, y = shuffle(X, y)
            local_step = 0
            state = None
            for X_batch, y_batch in zip(self.gen_batch(X, batch_size), self.gen_batch(y, batch_size)):
                varlen = np.random.choice([70, 75, 80, 85, 90], 1)[0]
                X_batch = tf.contrib.keras.preprocessing.sequence.pad_sequences(X_batch, maxlen=varlen)
                X_train_batch = torch.autograd.Variable(torch.from_numpy(X_batch.astype(np.int64)))
                y_train_batch = torch.autograd.Variable(torch.from_numpy(y_batch.astype(np.int64)))
                
                if (self.stateful) and (len(X_batch) == batch_size):
                    y_pred_batch, state = self.forward(X_train_batch, state)
                    state = (torch.autograd.Variable(state[0].data), torch.autograd.Variable(state[1].data))
                else:
                    y_pred_batch, _ = self.forward(X_train_batch)

                loss = self.criterion(y_pred_batch, y_train_batch)     # cross entropy loss
                self.optimizer, lr = self.adjust_lr(self.optimizer, global_step, total_steps)
                self.optimizer.zero_grad()                             # clear gradients for this training step
                loss.backward()                                        # backpropagation, compute gradients
                self.optimizer.step()                                  # apply gradients
                local_step += 1
                global_step += 1
                acc = (torch.max(y_pred_batch,1)[1].data.numpy().squeeze() == y_batch).mean()
                if local_step % 100 == 0:
                    print ('Epoch [%d/%d] | Step [%d/%d] | Loss: %.4f | Acc: %.4f | LR: %.4f'
                           %(epoch+1, n_epoch, local_step, n_batch, loss.data[0], acc, lr))
    # end method fit


    def evaluate(self, X_test, y_test, batch_size=32):
        correct = 0
        total = 0
        state = None
        for X_batch, y_batch in zip(self.gen_batch(X_test, batch_size), self.gen_batch(y_test, batch_size)):
            X_test_batch = torch.autograd.Variable(torch.from_numpy(X_batch.astype(np.int64)))
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
