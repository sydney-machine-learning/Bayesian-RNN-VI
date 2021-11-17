import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import time
import pickle

import sys
import os
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from BayesianMNN_data_processing_original import data_processing
import scipy.io as sio
import numpy as np
from sklearn.metrics import mean_squared_error

from torch.utils.data import DataLoader, Dataset

def extract_data(data):
    data = [data_processing(data[i], 4) for i in range(len(data))]
    X, Y = list(zip(*data))
    X = torch.tensor(np.concatenate(X, axis=0)).type(torch.float)
    Y = torch.tensor(np.concatenate(Y, axis=0)).type(torch.float)
    return X, Y

train = sio.loadmat('BBB_RNN\data\cytrack_train.mat')
train = train['cyclones_train']
train_final = train[0]

test = sio.loadmat('BBB_RNN\data\cytrack_test.mat')
test = test['cyclones_test']
test_final = test[0]

X_train, Y_train = extract_data(train_final)
X_test, Y_test = extract_data(test_final)



class data_set(Dataset):                                         #Custom dataset class for dataloader
    def __init__(self,trainx,labelx):
        data=[]
        for i in range(len(trainx)):
            data.append([trainx[i],labelx[i]])
        self.data = data
  
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, index):
        return self.data[index]

trainset = data_set(X_train,Y_train)
testset = data_set(X_test, Y_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024, shuffle=False) 
testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False) 