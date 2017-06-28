import torch


class RNNRegressor(torch.nn.Module):
    def __init__(self, n_in, n_out, cell_size):
        super(RNNRegressor, self).__init__()
        self.n_in = n_in
        self.cell_size = cell_size
        self.n_out = n_out
        self.build_model()
    # end constructor


    def build_model(self):
        self.lstm = torch.nn.GRU(self.n_in, self.cell_size, batch_first=True)
        self.fc = torch.nn.Linear(self.cell_size, self.n_out)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    # end method build_model    


    def forward(self, X, init_state):
        rnn_out, final_state = self.lstm(X, init_state)
        reshaped = rnn_out.contiguous().view(-1, self.cell_size)
        logits = self.fc(reshaped)
        return logits, final_state
    # end method forward    
# end class RNNRegressor
