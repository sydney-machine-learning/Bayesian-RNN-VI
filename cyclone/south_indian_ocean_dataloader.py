import csv
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

train_data_path = r'D:\python\BBB_RNN\data\southindianocean.csv'

IN_STEP = 5
OUT_STEP = 10

curr_x = []
            
with open(train_data_path, mode = 'r') as file:
    csvfile = csv.reader(file)
    train_X = []
    train_Y = []
    for i,lines in enumerate(csvfile):
        if i>=1 :
            if len(curr_x)>0 and int(lines[1])==curr_x[-1][0]:
               curr_x.append([int(lines[1]), float(lines[3]), float(lines[4])])
               if len(curr_x)>=IN_STEP+OUT_STEP:
                    x_temp, y_temp = [],[]
                    for j in range(-1*(IN_STEP+OUT_STEP),(-1*OUT_STEP)):
                        x_temp.append([curr_x[j][1], curr_x[j][2]])
                    for j in range(-1*OUT_STEP,0):
                        y_temp.append([curr_x[j][1], curr_x[j][2]])
                    train_X.append(x_temp)
                    train_Y.append(y_temp)
            elif len(curr_x)==0:
                curr_x.append([int(lines[1]), float(lines[3]), float(lines[4])])
            else:
                curr_x.clear()
                curr_x.append([int(lines[1]), float(lines[3]), float(lines[4])])
        if len(train_X)==1000:
            break


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




    
                 