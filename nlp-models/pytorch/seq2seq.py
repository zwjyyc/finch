from __future__ import print_function
from __future__ import division
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, encoder_embedding_dim, n_layers):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, encoder_embedding_dim)
        self.gru = nn.GRU(encoder_embedding_dim, hidden_size, batch_first=True)

    def forward(self, inputs, hidden):
        embedded = self.embedding(inputs)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        return result
# end class


class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, decoder_embedding_dim, n_layers):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, decoder_embedding_dim)
        self.gru = nn.GRU(decoder_embedding_dim, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, inputs, hidden):
        embedded = self.embedding(inputs)
        output, hidden = self.gru(embedded, hidden)
        output = self.out(output.view(-1, self.hidden_size))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        return result
# end class


class Seq2Seq:
    def __init__(self, rnn_size, n_layers,
                 X_word2idx, encoder_embedding_dim,
                 Y_word2idx, decoder_embedding_dim):
        self.X_word2idx = X_word2idx
        self.Y_word2idx = Y_word2idx

        self.encoder = EncoderRNN(len(self.X_word2idx), rnn_size, encoder_embedding_dim, n_layers)
        self.decoder = DecoderRNN(len(self.Y_word2idx), rnn_size, decoder_embedding_dim, n_layers)
        self.encoder_optimizer = optim.Adam(self.encoder.parameters())
        self.decoder_optimizer = optim.Adam(self.decoder.parameters())
        self.criterion = nn.CrossEntropyLoss()

        self.register_symbols()
    # end constructor


    def train(self, source, target):
        target_len = target.size()[1]

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        encoder_hidden = self.encoder.initHidden(source.size()[0])
        encoder_output, encoder_hidden = self.encoder(source, encoder_hidden)
        
        decoder_hidden = encoder_hidden

        decoder_input = self.process_decoder_input(target)
        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
        loss = self.criterion(decoder_output, target.view(-1))
        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.data[0] / target_len
    # end method


    def predict(self, source, maxlen=None):
        if maxlen is None:
            maxlen = 2 * source.size()[1]

        encoder_hidden = self.encoder.initHidden()
        encoder_output, encoder_hidden = self.encoder(source, encoder_hidden)
        
        decoder_input = Variable(torch.LongTensor([[self._y_go]]))
        decoder_hidden = encoder_hidden
        output_indices = []
        for i in range(maxlen):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_output = F.log_softmax(decoder_output)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            output_indices.append(ni)
            decoder_input = Variable(torch.LongTensor([[ni]]))
            if ni == self._y_eos:
                break
        return output_indices
    # end method


    def fit(self, X_train, Y_train, n_epoch=3, display_step=100, batch_size=1):
        for epoch in range(1, n_epoch+1):
            for local_step, (X_train_batch, Y_train_batch, X_train_batch_lens, Y_train_batch_lens) in enumerate(
                self.next_batch(X_train, Y_train, batch_size)):
                source = Variable(torch.from_numpy(X_train_batch.astype(np.int64)))
                target = Variable(torch.from_numpy(Y_train_batch.astype(np.int64)))
                loss = self.train(source, target)
                if local_step % display_step == 0:
                    print("Epoch %d/%d | Batch %d/%d | train_loss: %.3f |" % 
                          (epoch, n_epoch, local_step, len(X_train)//batch_size, loss))
                
    # end method


    def infer(self, input_word, X_idx2word, Y_idx2word):        
        input_indices = [self.X_word2idx.get(char, self._x_unk) for char in input_word]
        source = Variable(torch.from_numpy(np.atleast_2d(input_indices).astype(np.int64)))
        out_indices = self.predict(source)
        
        print('\nSource')
        print('Word: {}'.format([i for i in input_indices]))
        print('IN: {}'.format(' '.join([X_idx2word[i] for i in input_indices])))
        
        print('\nTarget')
        print('Word: {}'.format([i for i in out_indices]))
        print('OUT: {}'.format(' '.join([Y_idx2word[i] for i in out_indices])))
    # end method


    def pad_sentence_batch(self, sentence_batch, pad_int):
        padded_seqs = []
        seq_lens = []
        max_sentence_len = max([len(sentence) for sentence in sentence_batch])
        for sentence in sentence_batch:
            padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
            seq_lens.append(len(sentence))
        return padded_seqs, seq_lens
    # end method


    def next_batch(self, X, Y, batch_size, X_pad_int=None, Y_pad_int=None):
        if X_pad_int is None:
            X_pad_int = self._x_pad
        if Y_pad_int is None:
            Y_pad_int = self._y_pad
        
        for i in range(0, len(X) - len(X) % batch_size, batch_size):
            X_batch = X[i : i + batch_size]
            Y_batch = Y[i : i + batch_size]
            padded_X_batch, X_batch_lens = self.pad_sentence_batch(X_batch, X_pad_int)
            padded_Y_batch, Y_batch_lens = self.pad_sentence_batch(Y_batch, Y_pad_int)
            yield (np.array(padded_X_batch),
                   np.array(padded_Y_batch),
                   X_batch_lens,
                   Y_batch_lens)
    # end method


    def register_symbols(self):
        self._x_go = self.X_word2idx['<GO>']
        self._x_eos = self.X_word2idx['<EOS>']
        self._x_pad = self.X_word2idx['<PAD>']
        self._x_unk = self.X_word2idx['<UNK>']

        self._y_go = self.Y_word2idx['<GO>']
        self._y_eos = self.Y_word2idx['<EOS>']
        self._y_pad = self.Y_word2idx['<PAD>']
        self._y_unk = self.Y_word2idx['<UNK>']
    # end method


    def process_decoder_input(self, target):
        target = target.data.numpy()
        main = target[:, :-1]
        decoder_input = np.concatenate([np.full([1, 1], self._y_go), main], 1)
        return Variable(torch.from_numpy(decoder_input.astype(np.int64)))
    # end method    
# end class
