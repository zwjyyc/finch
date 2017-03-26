import torchvision.datasets as dsets
import torchvision.transforms as transforms
from rnn_clf import RNNClassifier


# Hyper Parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

# MNIST Dataset
train_dataset = dsets.MNIST(root='../data/',
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='../data/',
                           train=False, 
                           transform=transforms.ToTensor())
"""
# Data Loader (Input Pipeline)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)
"""
rnn = RNNClassifier(input_size, sequence_length, hidden_size, num_layers, num_classes)
rnn.fit(train_dataset, num_epochs, batch_size)
    
