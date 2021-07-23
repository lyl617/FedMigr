__author__ = 'yang.xu'

import os
import time
import math
import numpy as np
import numpy.ma as ma
import random
import re
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.nn as nn
#from syft.frameworks.torch.nn import GRU, LSTM, RNN
from torch.utils.data import TensorDataset, DataLoader

import syft as sy  # <--NEW: import the PySyft library
import syft.frameworks.torch.nn as syft_nn

# <--For Sent140
'''
an automatically generated sentiment analysis dataset that annotates tweets
based on the emoticons present in them. Each device is a different twitter user
'''
class Sent140_GRU(nn.Module):
    #'sent140.stacked_lstm': (0.0003, 25, 2, 100), # lr, seq_len, num_classes, num_hidden
    def __init__(self, input_size=300, hidden_size=128, num_layers=1, num_classes=2):
        super(Sent140_GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.rnn = syft_nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first = False)
        self.out = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x, hidden=None):
        x, _ = self.rnn(x, hidden)
        x = F.relu(x[:,-1,:], inplace=True)
        x = self.out(x)
        return F.log_softmax(x, dim=1)

# class Sent140_RNN(nn.Module):
#     def __init__(self, input_size=300, hidden_size=128, output_size=2):
#         super(Sent140_RNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
#         self.i2o = nn.Linear(input_size + hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=1)

#     def forward(self, input, hidden):
#         combined = torch.cat((input, hidden), 1)
#         hidden = self.i2h(combined)
#         output = self.i2o(combined)
#         output = self.softmax(output)
#         return output, hidden

#     def initHidden(self):
#         return torch.zeros(1, self.hidden_size)

def test_GRU():
    """
    Test the GRU module to ensure that it produces the exact same
    output as the primary torch implementation, in the same order.
    """

    # Disable mkldnn to avoid rounding errors due to difference in implementation
    mkldnn_enabled_init = torch._C._get_mkldnn_enabled()
    torch._C._set_mkldnn_enabled(False)

    batch_size = 50
    input_size = 300
    hidden_size = 100
    num_layers = 2
    seq_len = 25

    test_input = torch.rand(seq_len, batch_size, input_size)
    test_hidden_state = torch.rand(num_layers, batch_size, hidden_size)

    # GRU implemented in pysyft
    rnn_syft = syft_nn.GRU(input_size, hidden_size, num_layers)

    # GRU implemented in original pytorch
    rnn_torch = nn.GRU(input_size, hidden_size, num_layers)

    # Make sure the weights of both GRU are identical
    rnn_syft.rnn_forward[0].fc_xh.weight = rnn_torch.weight_ih_l0
    rnn_syft.rnn_forward[0].fc_xh.bias = rnn_torch.bias_ih_l0
    rnn_syft.rnn_forward[0].fc_hh.weight = rnn_torch.weight_hh_l0
    rnn_syft.rnn_forward[0].fc_hh.bias = rnn_torch.bias_hh_l0

    output_syft, hidden_syft = rnn_syft(test_input, test_hidden_state)
    output_torch, hidden_torch = rnn_torch(test_input, test_hidden_state)

    # Reset mkldnn to the original state
    torch._C._set_mkldnn_enabled(mkldnn_enabled_init)

    # Assert the hidden_state and output of both models are identical separately
    assert torch.all(torch.lt(torch.abs(output_syft - output_torch), 1e-6))
    assert torch.all(torch.lt(torch.abs(hidden_syft - hidden_torch), 1e-6))

# <--For CIFAR10

class VGG9(nn.Module):
    def __init__(self):
        super(VGG9, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self._initialize_weights()

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            #nn.Linear(4096, 1024),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            #nn.Linear(1024, 512),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return F.log_softmax(x, dim=1)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

# <--For CIFAR10

class CIFAR10_Net(nn.Module):
    def __init__(self):
        super(CIFAR10_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x), inplace=True))
        x = self.pool(F.relu(self.conv2(x), inplace=True))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CIFAR10_Deep_Net(nn.Module):
    def __init__(self):
        super(CIFAR10_Deep_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x), inplace=True))
        x = self.pool(F.relu(self.conv2(x), inplace=True))
        x = x.view(-1, 128 * 5 * 5)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


# <--For FashionMNIST & MNIST
class MNIST_Small_Net(nn.Module):
    def __init__(self):
        super(MNIST_Small_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 32, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 32, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 32)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 64)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class MNIST_LR_Net(nn.Module):
    def __init__(self):
        super(MNIST_LR_Net, self).__init__()
        self.hidden1 = nn.Linear(28 * 28, 512)
        self.hidden2 = nn.Linear(512, 512)
        self.out = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.hidden1(x), inplace=True)
        x = F.relu(self.hidden2(x), inplace=True)
        x = self.out(x)
        return F.log_softmax(x, dim=1)
        
class CIFAR100_Net(nn.Module):
    def __init__(self):
        super(CIFAR100_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 100)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x), inplace=True))
        x = self.pool(F.relu(self.conv2(x), inplace=True))
        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def create_model_instance(dataset_type, model_type):
    if dataset_type == 'CIFAR10':
        if model_type == 'VGG':
            model = VGG9()
    if dataset_type == 'CIFAR100':
        if model_type == 'ResNet':
            model = ResNet9(num_classes=100)

    return model