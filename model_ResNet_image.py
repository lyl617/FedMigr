import torch
import sys
import time
from torchvision import datasets, transforms
from torch import nn, optim
import numpy as np
import torchvision
import torch.nn.functional as F
import random
import os
import copy



#the defination of VGG16, including 22 layer

'''
models = [
    nn.Sequential(
            torch.nn.Conv2d(3, 96, 11, 4, 0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,2)
        ),
    nn.Sequential(
            torch.nn.Conv2d(96, 256, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,2)
        ),
    nn.Sequential(
            torch.nn.Conv2d(256,384, 3, 1, 1),
            torch.nn.ReLU(),
        ),
    nn.Sequential(
            torch.nn.Conv2d(384,384, 3, 1, 1),
            torch.nn.ReLU(),
        ),
    nn.Sequential(
            torch.nn.Conv2d(384,256, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,2)
        ),
    nn.Sequential(
            torch.nn.Linear(9216, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
    ),
    nn.Sequential(
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
    ),
    nn.Sequential(
            torch.nn.Linear(4096, 1000)
    )
]
'''
class layer0(nn.Module):
    def __init__(self):
        super(layer0, self).__init__()
        self.conv = nn.Conv2d(3, 96, 11,4,0)
        self.BN1 = nn.BatchNorm2d(96)
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.BN1(x)
        return x

class layer1(nn.Module):
    def __init__(self):
        super(layer1,self).__init__()
        self.pool = nn.MaxPool2d(3,2)
    def forward(self,x):
        x = self.pool(x)
        return x

class layer2(nn.Module):
    def __init__(self):
        super(layer2, self).__init__()
        self.conv = nn.Conv2d(96, 256, 5, 1, 2)
        self.BN1 = nn.BatchNorm2d(256)
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.BN1(x)
        return x

class layer3(nn.Module):
    def __init__(self):
        super(layer3,self).__init__()
        self.pool = nn.MaxPool2d(3,2)
    def forward(self,x):
        x = self.pool(x)
        return x


class layer4(nn.Module):
    def __init__(self):
        super(layer4, self).__init__()
        self.conv = nn.Conv2d(256,384, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(384)
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        return x

class layer5(nn.Module):
    def __init__(self):
        super(layer5, self).__init__()
        self.conv = nn.Conv2d(384,384, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(384)
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.BN1(x)
        return x

class layer6(nn.Module):
    def __init__(self):
        super(layer6, self).__init__()
        self.conv = nn.Conv2d(384,256, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(256)
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.BN1(x)
        return x

class layer7(nn.Module):
    def __init__(self):
        super(layer7,self).__init__()
        self.pool = nn.MaxPool2d(3,2)
    def forward(self,x):
        x = self.pool(x)
        x = x.view(-1,9216)
        return x

class layer8(nn.Module):
    def __init__(self):
        super(layer8,self).__init__()
        self.fc = nn.Linear(9216, 4096)
        self.drop = nn.Dropout(0.5)
    def forward(self,x):
        x = F.relu(self.fc(x), inplace=True)
        x = self.drop(x)
        return x

class layer9(nn.Module):
    def __init__(self):
        super(layer9,self).__init__()
        self.fc = nn.Linear(4096, 4096)
        self.drop = nn.Dropout()
    def forward(self,x):
        x = F.relu(self.fc(x), inplace=True)
        x = self.drop(x)
        return x

class layer10(nn.Module):
    def __init__(self):
        super(layer10,self).__init__()
        self.fc = nn.Linear(4096, 10)
    def forward(self,x):
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def construct_AlexNet_image(partition_way, lr):
    models=[]
    optimizers=[]
    for i in range(0,len(partition_way)):
        if i==0:
            if partition_way[i] == 0:
                model = layer0()
                optimizer = optim.SGD(params = model.parameters(), lr = lr, momentum = 0.9)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==1:
            if partition_way[i] == 0:
                model = layer1()
            else:
                model = None
            optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==2:
            if partition_way[i] == 0:
                model = layer2()
                optimizer = optim.SGD(params = model.parameters(), lr = lr, momentum = 0.9)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==3:
            if partition_way[i] == 0:
                model = layer3()
            else:
                model = None
            optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==4:
            if partition_way[i] == 0:
                model = layer4()
                optimizer = optim.SGD(params = model.parameters(), lr = lr, momentum = 0.9)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==5:
            if partition_way[i] == 0:
                model = layer5()
                optimizer = optim.SGD(params = model.parameters(), lr = lr, momentum = 0.9)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==6:
            if partition_way[i] == 0:
                model = layer6()
                optimizer = optim.SGD(params = model.parameters(), lr = lr, momentum = 0.9)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==7:
            if partition_way[i] == 0:
                model = layer7()
            else:
                model = None
            optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==8:
            if partition_way[i] == 0:
                model = layer8()
                optimizer = optim.SGD(params = model.parameters(), lr = lr, momentum = 0.9)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==9:
            if partition_way[i] == 0:
                model = layer9()
                optimizer = optim.SGD(params = model.parameters(), lr = lr, momentum = 0.9)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==10:
            if partition_way[i] == 0:
                model = layer10()
                optimizer = optim.SGD(params = model.parameters(), lr = lr, momentum = 0.9)
            else:
                model = None
                optimizer = None
            models.append(model)
            optimizers.append(optimizer)
    return models, optimizers