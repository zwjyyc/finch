import torch 
import torch.nn as nn
from torch.autograd import Variable


class RNNClassifier(nn.Module):
    def __init__(self, input_size, sequence_length, hidden_size, num_layers, num_classes):
        super(RNNClassifier, self).__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
        self.build_model()
    
    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) # set initial states  
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) # set initial states 
        
        out, _ = self.lstm(x, (h0, c0)) # forward propagate
        
        out = self.fc(out[:, -1, :]) # decode hidden state of last time step
        return out

    def build_model(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def fit(self, train_dataset, num_epochs, batch_size):
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size, 
                                                   shuffle=True)

        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = Variable(images.view(-1, self.sequence_length, self.input_size))
                labels = Variable(labels)
                
                # forward + backward + optimize
                self.optimizer.zero_grad()
                outputs = self.forward(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                if (i+1) % 100 == 0:
                    print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                           %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

