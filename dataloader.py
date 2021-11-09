import csv
from os import sep
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

# Constants
data_path = r'data/Sunspot/train.txt'
IN_STEP = 4
OUT_STEP = 1

class SunspotDataset(Dataset):
    """Custom dataset class for Sunspot Time-series dataset"""
    def __init__(self, X, y):
        assert(len(X) == len(y)), "X and y must have same length"
        self.X = X
        self.y = y
  
    def __len__(self):
        return len(self.X)
  
    def __getitem__(self, index):
        return (self.X[index], self.y[index])


# Read data from csv file
data = np.loadtxt(data_path, delimiter=" ")

# Split Input and Output
train_X = data[:, :IN_STEP].astype(np.float32)
train_Y = data[:, IN_STEP:IN_STEP+OUT_STEP].astype(np.float32)

# DataLoader
train_X = torch.Tensor(train_X) 
train_Y = torch.Tensor(train_Y)
train_dataset = SunspotDataset(train_X, train_Y)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)


# Read data from csv file
test_data_path = r'data/Sunspot/test.txt'
test_data = np.loadtxt(test_data_path, delimiter=" ")

# Split Input and Output
test_X = data[:, :IN_STEP].astype(np.float32)
test_Y = data[:, IN_STEP:IN_STEP+OUT_STEP].astype(np.float32)

# DataLoader
test_X = torch.Tensor(test_X) 
test_Y = torch.Tensor(test_Y)
test_dataset = SunspotDataset(test_X, test_Y)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)