import torch
import numpy as np
import math
from sklearn.utils import shuffle


class RNNTextGen(torch.nn.Module):
    def __init__(self, text, seq_len=50, embedding_dim=128, cell_size=256, n_layer=2, stateful=True):
        super(RNNTextGen, self).__init__()
        self.text = text
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.cell_size = cell_size
        self.n_layer = n_layer
        self.stateful = stateful
        self.preprocessing()
        self.build_model()
    # end constructor


    def build_model(self):
        self.encoder = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = torch.nn.LSTM(self.embedding_dim, self.cell_size, self.n_layer, batch_first=True)
        self.fc = torch.nn.Linear(self.cell_size, self.vocab_size)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters())
    # end method build_model    


    def forward(self, X, init_state=None):
        rnn_out, final_state = self.lstm(self.encoder(X), init_state)
        logits = self.fc(rnn_out.contiguous().view(-1, self.cell_size))
        return logits, final_state
    # end method forward


    def fit(self, start_word, n_gen=500, text_iter_step=1, n_epoch=1, batch_size=128):
        global_step = 0
        n_batch = (len(self.indexed) - self.seq_len*batch_size - 1) // text_iter_step
        total_steps = n_epoch * n_batch

        for epoch in range(n_epoch):
            state = None
            for local_step, (X_batch, Y_batch) in enumerate(self.next_batch(batch_size, text_iter_step)):
                inputs = torch.autograd.Variable(torch.from_numpy(X_batch.astype(np.int64)))
                labels = torch.autograd.Variable(torch.from_numpy(Y_batch.astype(np.int64)))
                
                if (self.stateful) and (len(X_batch) == batch_size):
                    preds, state = self.forward(inputs, state)
                    state = (torch.autograd.Variable(state[0].data), torch.autograd.Variable(state[1].data))
                else:
                    preds, _ = self.forward(inputs)

                loss = self.criterion(preds.view(-1, self.vocab_size), labels.view(-1))
                self.optimizer.zero_grad()                             # clear gradients for this training step
                loss.backward()                                        # backpropagation, compute gradients
                self.optimizer.step()                                  # apply gradients
                global_step += 1

                if local_step % 10 == 0:
                    print ('Epoch [%d/%d] | Step [%d/%d] | Loss: %.4f'
                           %(epoch+1, n_epoch, local_step, n_batch, loss.data[0]))
                if local_step % 100 == 0:
                    print(self.infer(start_word, n_gen)+'\n')
    # end method fit


    def infer(self, start_word, n_gen):
        # warming up
        state = None
        char_list = list(start_word)
        for char in char_list[:-1]:
            x = np.atleast_2d(self.char2idx[char])
            input = torch.autograd.Variable(torch.from_numpy(x.astype(np.int64))) 
            _, state = self.forward(input, state)
            state = (torch.autograd.Variable(state[0].data), torch.autograd.Variable(state[1].data))
        # end warming up

        out_sentence = 'IN:\n' + start_word + '\n\nOUT:\n' + start_word
        char = char_list[-1]
        for _ in range(n_gen):
            x = np.atleast_2d(self.char2idx[char])
            input = torch.autograd.Variable(torch.from_numpy(x.astype(np.int64))) 
            logits, state = self.forward(input, state)
            softmax_out = torch.nn.functional.softmax(logits)
            probas = softmax_out.data.numpy()[0].astype(np.float64)
            probas = probas / np.sum(probas)
            actions = np.random.multinomial(1, probas, 1)
            char = self.idx2char[np.argmax(actions)]
            out_sentence = out_sentence + char
        return out_sentence
    # end method infer


    def preprocessing(self):
        text = self.text 
        chars = set(text)
        self.char2idx = {c: i for i, c in enumerate(chars)}
        self.idx2char = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(self.idx2char)
        print('Vocabulary size:', self.vocab_size)

        self.indexed = np.array([self.char2idx[char] for char in list(text)])
    # end method text_preprocessing


    def next_batch(self, batch_size, text_iter_step):
        window = self.seq_len * batch_size
        for i in range(0, len(self.indexed)-window-1, text_iter_step):
            yield (self.indexed[i : i+window].reshape(-1, self.seq_len),
                   self.indexed[i+1 : i+window+1].reshape(-1, self.seq_len))
    # end method next_batch
# end class