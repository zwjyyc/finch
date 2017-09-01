from __future__ import print_function
from __future__ import division
import numpy as np
import torch
from extras import nll


class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, encoder_embedding_dim, n_layers):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder_embedding_dim = encoder_embedding_dim
        self.n_layers = n_layers
        self.build_model()
    # end constructor


    def build_model(self):
        self.embedding = torch.nn.Embedding(self.input_size, self.encoder_embedding_dim)
        self.lstm = torch.nn.LSTM(self.encoder_embedding_dim, self.hidden_size,
                                  batch_first=True, num_layers=self.n_layers) 
    # end method


    def forward(self, inputs, hidden, X_lens):
        embedded = self.embedding(inputs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, X_lens, batch_first=True)
        rnn_out, hidden = self.lstm(packed, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        return output, hidden
    # end method


    def init_hidden(self, batch_size):
        result = (torch.autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                  torch.autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
        return result
    # end method
# end class


class Decoder(torch.nn.Module):
    def __init__(self, output_size, hidden_size, decoder_embedding_dim, n_layers):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.decoder_embedding_dim = decoder_embedding_dim
        self.n_layers = n_layers
        self.build_model()
    # end constructor


    def build_model(self):
        self.embedding = torch.nn.Embedding(self.output_size, self.decoder_embedding_dim)
        self.lstm = torch.nn.LSTM(self.decoder_embedding_dim, self.hidden_size,
                                  batch_first=True, num_layers=self.n_layers)
        self.out = torch.nn.Linear(self.hidden_size, self.output_size)
    # end method


    def forward(self, inputs, hidden):
        embedded = self.embedding(inputs)
        output, hidden = self.lstm(embedded, hidden)
        output = self.out(output.contiguous().view(-1, self.hidden_size))
        return output, hidden
    # end method
# end class


class Seq2Seq:
    def __init__(self, rnn_size, n_layers,
                 X_word2idx, encoder_embedding_dim,
                 Y_word2idx, decoder_embedding_dim, max_grad_norm=5.0):
        self.rnn_size = rnn_size
        self.n_layers = n_layers
        self.X_word2idx = X_word2idx
        self.Y_word2idx = Y_word2idx
        self.encoder_embedding_dim = encoder_embedding_dim
        self.decoder_embedding_dim = decoder_embedding_dim
        self.max_grad_norm = max_grad_norm
        self.build_model()
        self.register_symbols()
    # end constructor


    def build_model(self):
        self.encoder = Encoder(len(self.X_word2idx), self.rnn_size, self.encoder_embedding_dim, self.n_layers)
        self.decoder = Decoder(len(self.Y_word2idx), self.rnn_size, self.decoder_embedding_dim, self.n_layers)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters())
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters())
        self.criterion = torch.nn.CrossEntropyLoss()
    # end method


    def train(self, source, target, X_lens, Y_masks):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        encoder_hidden = self.encoder.init_hidden(source.size(0))
        encoder_output, encoder_hidden = self.encoder(source, encoder_hidden, X_lens)
        
        decoder_hidden = encoder_hidden

        decoder_input = self.process_decoder_input(target)
        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

        losses = nll(torch.nn.functional.log_softmax(decoder_output), target.view(-1, 1))
	Y_masks = torch.autograd.Variable(torch.FloatTensor(Y_masks)).view(-1)
	loss = torch.mul(losses, Y_masks).sum() / source.size(0)
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.encoder.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm(self.decoder.parameters(), self.max_grad_norm)
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.data[0] / (target.size(1))
    # end method


    def predict(self, source, maxlen=None):
        if maxlen is None:
            maxlen = 2 * source.size(1)

        encoder_hidden = self.encoder.init_hidden(1)
        encoder_output, encoder_hidden = self.encoder(source, encoder_hidden, [source.size()[1]])
        
        decoder_hidden = encoder_hidden        

        decoder_input = torch.autograd.Variable(torch.LongTensor([[self._y_go]]))
        output_indices = []
        for i in range(maxlen):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_output = torch.nn.functional.log_softmax(decoder_output)
            topv, topi = decoder_output.data.topk(1)
            topi = topi[0][0]
            output_indices.append(topi)
            decoder_input = torch.autograd.Variable(torch.LongTensor([[topi]]))
            if topi == self._y_eos:
                break
        return output_indices
    # end method


    def fit(self, X_train, Y_train, n_epoch=60, display_step=100, batch_size=128):
        X_train, Y_train = self.sort(X_train, Y_train)
        for epoch in range(1, n_epoch+1):
            for local_step, (X_train_batch, Y_train_batch, X_train_batch_lens, Y_train_batch_masks) in enumerate(
                self.next_batch(X_train, Y_train, batch_size)):
                source = torch.autograd.Variable(torch.from_numpy(X_train_batch.astype(np.int64)))
                target = torch.autograd.Variable(torch.from_numpy(Y_train_batch.astype(np.int64)))
                loss = self.train(source, target, X_train_batch_lens, Y_train_batch_masks)
                if local_step % display_step == 0:
                    print("Epoch %d/%d | Batch %d/%d | train_loss: %.3f |" % 
                          (epoch, n_epoch, local_step, len(X_train)//batch_size, loss))       
    # end method


    def infer(self, input_word, X_idx2word, Y_idx2word):        
        input_indices = [self.X_word2idx.get(char, self._x_unk) for char in input_word]
        source = torch.autograd.Variable(torch.from_numpy(np.atleast_2d(input_indices).astype(np.int64)))
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
	masks = []
        max_sentence_len = max([len(sentence) for sentence in sentence_batch])
        for sentence in sentence_batch:
            padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
            seq_lens.append(len(sentence))
            masks.append([1] * len(sentence) + [0] * (max_sentence_len - len(sentence)))
        return padded_seqs, seq_lens, masks
    # end method


    def next_batch(self, X, Y, batch_size, X_pad_int=None, Y_pad_int=None):
        if X_pad_int is None:
            X_pad_int = self._x_pad
        if Y_pad_int is None:
            Y_pad_int = self._y_pad
        
        for i in range(0, len(X) - len(X) % batch_size, batch_size):
            X_batch = X[i : i + batch_size]
            Y_batch = Y[i : i + batch_size]
            padded_X_batch, X_batch_lens, _ = self.pad_sentence_batch(X_batch, X_pad_int)
            padded_Y_batch, _, Y_batch_masks = self.pad_sentence_batch(Y_batch, Y_pad_int)
            yield (np.array(padded_X_batch),
                   np.array(padded_Y_batch),
                   X_batch_lens,
                   Y_batch_masks)
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
        target = target[:, :-1]
	go = torch.autograd.Variable((torch.zeros(target.size(0), 1) + self._y_go).long())
        decoder_input = torch.cat((go, target), 1)
        return decoder_input
    # end method


    def sort(self, X, Y):
        lens = [len(x) for x in X]
        idx = list(reversed(np.argsort(lens)))
        X = np.array(X)[idx].tolist()
        Y = np.array(Y)[idx].tolist()
        return X, Y
    # end method    
# end class
