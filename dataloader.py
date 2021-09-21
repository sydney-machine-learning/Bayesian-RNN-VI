import csv
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

data_path = r'D:\python\BBB_RNN\train1.csv'
OUT_STEP = 5

with open(data_path, mode = 'r') as file:
    csvfile = csv.reader(file)
    train_X = []
    train_Y = []
    for i,lines in enumerate(csvfile):
        if i>1:
           x_temp = []
           for j in range(1,6):
               x_temp.append(float(lines[j]))
           y_temp=[]
           ind_end = 6+OUT_STEP
           for j in range(6,ind_end):
               y_temp.append(float(lines[j]))
           train_X.append(x_temp)
           train_Y.append(y_temp)

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
train_X = torch.Tensor(train_X) 
train_Y = torch.Tensor(train_Y)
trainset = data_set(train_X,train_Y)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=False) 


    
                 