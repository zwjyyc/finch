import mxnet as mx
import numpy as np


class RNNTextGen(mx.gluon.Block):
    def __init__(self, ctx, text, seq_len=50, embedding_dim=128, rnn_size=256, n_layer=2, lr=1e-3):
        super(RNNTextGen, self).__init__()
        self.ctx = ctx
        self.text = text
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.rnn_size = rnn_size
        self.n_layer = n_layer
        self.lr = lr

        self.preprocessing()
        self.build_model()
        self.compile_model()
    # end constructor


    def build_model(self):
        self.encoder = mx.gluon.nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = mx.gluon.rnn.LSTM(self.rnn_size, self.n_layer, 'NTC')
        self.output = mx.gluon.nn.Dense(self.vocab_size)
    # end method


    def forward(self, x, hidden):
        rnn_out, hidden = self.lstm(self.encoder(x), hidden)
        logits = self.output(rnn_out.reshape((-1, self.rnn_size)))
        return logits, hidden
    # end method
        

    def compile_model(self):
        self.collect_params().initialize(mx.init.Xavier(), ctx=self.ctx)
        self.criterion = mx.gluon.loss.SoftmaxCrossEntropyLoss()
        self.optimizer = mx.gluon.Trainer(self.collect_params(), 'adam', {'learning_rate': self.lr})
    # end method


    def fit(self, start_word, n_gen=500, text_iter_step=1, n_epoch=1, batch_size=128):
        n_batch = (len(self.indexed) - self.seq_len*batch_size - 1) // text_iter_step
        for epoch in range(n_epoch):
            hidden = [mx.nd.zeros((self.n_layer, batch_size, self.rnn_size), self.ctx)] * 2
            for local_step, (X_batch, Y_batch) in enumerate(self.next_batch(batch_size, text_iter_step)):
                inputs, labels = self.from_numpy(X_batch, Y_batch)   
                inputs = inputs.as_in_context(self.ctx)
                labels = labels.as_in_context(self.ctx)
                with mx.gluon.autograd.record(train_mode=True):
                    output, hidden = self.forward(inputs, hidden)
                    hidden = self.detach(hidden)
                    loss = self.criterion(output, labels.reshape([-1]))
                loss.backward()
                self.optimizer.step(batch_size)
                loss = mx.nd.mean(loss).asscalar()
                if local_step % 10 == 0:
                    print ('Epoch [%d/%d] | Step [%d/%d] | Loss: %.4f'
                           %(epoch+1, n_epoch, local_step, n_batch, loss))
                if local_step % 50 == 0:
                    print(self.infer(start_word, n_gen)+'\n')
    # end method fit


    def infer(self, start_word, n_gen):
        # warming up
        hidden = [mx.nd.zeros((self.n_layer, 1, self.rnn_size), self.ctx)] * 2
        char_list = list(start_word)
        for char in char_list[:-1]:
            x = self.from_numpy(np.atleast_2d(self.char2idx[char])) 
            _, hidden = self.forward(x, hidden)

        out_sentence = 'IN:\n' + start_word + '\n\nOUT:\n' + start_word
        char = char_list[-1]
        for _ in range(n_gen):
            x = self.from_numpy(np.atleast_2d(self.char2idx[char]))
            logits, hidden = self.forward(x, hidden)
            softmax_out = mx.nd.softmax(logits)
            probas = softmax_out.asnumpy()[0].astype(np.float64)
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
    # end method


    def next_batch(self, batch_size, text_iter_step):
        window = self.seq_len * batch_size
        for i in range(0, len(self.indexed)-window-1, text_iter_step):
            yield (self.indexed[i : i+window].reshape(-1, self.seq_len),
                   self.indexed[i+1 : i+window+1].reshape(-1, self.seq_len))
    # end method


    def from_numpy(self, *args):
        data = []
        for _arr in args:
            arr = mx.nd.zeros(_arr.shape)
            arr[:] = _arr
            data.append(arr)
        return data if len(data) > 1 else data[0]
    # end method


    def detach(self, arrs):
        return [arr.detach() for arr in arrs]
    # end method
# end class
