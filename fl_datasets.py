__author__ = 'yang.xu'

#Import required libraries
import torch
import numpy as np
from torch.utils.data import Dataset
# import syft as sy
from torchvision import datasets, transforms
import fl_utils

class train_test_dataset():
    def __init__(self, data, targets, num_classes):
        self.data = data
        self.targets = targets
        self.classes = num_classes


class VMDataset(Dataset):

    def __init__(self, data, labels, transform = transforms.ToTensor()):
        
        """Args:
             
             images (Numpy Array): Data
             labels (Numpy Array): Labels corresponding to each data item
             transform (Optional): If any torch transform has to be performed on the dataset
             
        """
        
        "Attributes self.data and self.targets must be initialized."
        
        #<--Data must be initialized as self.data, self.train_data or self.test_data
        self.data = data
        #<--Targets must be initialized as self.targets, self.test_labels or self.train_labels
        self.targets = labels
        
        #<--The data and target must be converted to torch tensors before it is returned by __getitem__ method
        self.to_torchtensor()
        
        #<--If any transforms have to be performed on the dataset
        self.transform = transform
        
        
    def to_torchtensor(self):
        
        "Transform Numpy Arrays to Torch tensors."
        
        self.data = torch.from_numpy(self.data)
        self.labels = torch.from_numpy(self.targets)
    
        
    def __len__(self):
        
        """Required Method
            
           Returns:
        
                Length [int]: Length of Dataset/batches
        
        """
        
        return len(self.data)
    

    def __getitem__(self, idx):
        
        """Required Method
        
           The output of this method must be torch tensors since torch tensors are overloaded 
           with share() method which is used to share data to workers.
        
           Args:
                 
                 idx [integer]: The index of required batch/example
                 
           Returns:
                 
                 Data [Torch Tensor]:     The training examples
                 Target [ Torch Tensor]:  Corresponding labels of training examples 
        
        """
        
        sample = self.data[idx]
        target = self.targets[idx]
        # print('--[Debug] sample[0][0]:',sample[0][0])
        if self.transform:
            sample = self.transform(sample)

        #print('--[Debug] sample[0][0]:',sample[0][0])
        return sample, target

def load_datasets(dataset_type):
    train_dataset = []
    test_dataset = []
    
    if dataset_type == 'CIFAR10':
        train_dataset = datasets.CIFAR10('../data', train = True, download = False, transform = None)

        test_dataset = datasets.CIFAR10('../data', train = False, transform = None)

    elif dataset_type == 'CIFAR100':
        train_dataset = datasets.CIFAR100('../data', train = True, download = True, transform = None)
        
        test_dataset = datasets.CIFAR100('../data', train = False, transform = None)

    elif dataset_type == 'FashionMNIST':
        train_dataset = datasets.FashionMNIST('../data', train = True, download = True, transform = None)
    
        test_dataset = datasets.FashionMNIST('../data', train = False, transform = None)
    
    elif dataset_type == 'MNIST':
        train_dataset = datasets.MNIST('../data', train = True, download = True, transform = None)
    
        test_dataset = datasets.MNIST('../data', train = False, transform = None)
        
    elif dataset_type == 'Sent140':
        vocab_dir = '../data/sent140/embs.json'
        train_data_dir = '../data/sent140/data/train' # 254,555 users, 908,652 samples, 2 classes
        test_data_dir = '../data/sent140/data/test' # 286,281 samples, 2 classes
        word_emb_arr, ind_dict, _ = fl_utils.load_word_emb(vocab_dir)
        users, _, train_dataset_tmp, test_dataset_tmp = fl_utils.read_data(train_data_dir, test_data_dir)

        # train dataset
        train_dataset_tmp = [train_dataset_tmp[u] for u in users]
        data = [fl_utils.process_x(d['x'], ind_dict, word_emb_arr) for d in train_dataset_tmp]
        data = np.vstack(data) # array, (num_samples, max_words, emb_dim)

        targets_tmp  = [d['y'] for d in train_dataset_tmp]
        targets=[]
        [targets.extend(t) for t in targets_tmp]
        num_classes = len(set(targets))
        targets = np.array(targets) # array, (num_samples,)
        
        train_dataset = train_test_dataset(data, targets, num_classes)

        # test dataset
        test_dataset_tmp = [test_dataset_tmp[u] for u in users]
        data = [fl_utils.process_x(d['x'], ind_dict, word_emb_arr) for d in test_dataset_tmp]
        data = np.vstack(data) # array, (num_samples, max_words, emb_dim)

        targets_tmp  = [d['y'] for d in test_dataset_tmp]
        targets=[]
        [targets.extend(t) for t in targets_tmp]
        targets = np.array(targets) # array, (num_samples,)

        test_dataset = train_test_dataset(data, targets, num_classes)


        del data
        del targets
        del targets_tmp
        del train_dataset_tmp
        del test_dataset_tmp

    return train_dataset, test_dataset

def load_default_transform(dataset_type):
    dataset_transform = []
    
    if dataset_type == 'CIFAR10':
        dataset_transform = transforms.Compose([
                           transforms.ToPILImage(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ])

    elif dataset_type == 'CIFAR100':
        dataset_transform = transforms.Compose([
                           transforms.ToPILImage(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ])

    elif dataset_type == 'FashionMNIST':
          dataset_transform=transforms.Compose([
                           transforms.ToPILImage(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                         ])
    
    elif dataset_type == 'MNIST':
          dataset_transform=transforms.Compose([
                           transforms.ToPILImage(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                         ])
    elif dataset_type == 'Sent140':
          dataset_transform=None
                       
    return dataset_transform


def load_customized_transform(dataset_type):
    dataset_transform = []
    
    if dataset_type == 'CIFAR10':
        dataset_transform = transforms.Compose([
                           transforms.ToPILImage(),
                           transforms.RandomHorizontalFlip(0.6),
                           transforms.ToTensor()
                         ])

    elif dataset_type == 'CIFAR100':
          dataset_transform = transforms.Compose([
                           transforms.ToPILImage(),
                           transforms.RandomHorizontalFlip(1.0),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                         ])

    elif dataset_type == 'FashionMNIST':
          dataset_transform = transforms.Compose([
                           transforms.ToPILImage(),
                           transforms.RandomHorizontalFlip(1.0),
                           transforms.ToTensor()
                         ])
    
    elif dataset_type == 'MNIST':
          dataset_transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                         ])
    elif dataset_type == 'Sent140':
          dataset_transform=None
                       
    return dataset_transform