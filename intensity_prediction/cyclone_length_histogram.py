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
    
from data_processing import data_processing
import scipy.io as sio
import numpy as np
from sklearn.metrics import mean_squared_error

from torch.utils.data import DataLoader, Dataset

def get_number_of_cyclones(cyclone_name):
    train_data = os.path.join(os.getcwd(), 'data',cyclone_name, f'new_{cyclone_name}_train.mat')
    train = sio.loadmat(train_data)
    train = train['cyclones_train']
    train_final = train[0]

    test_data = os.path.join(os.getcwd(), 'data',cyclone_name, f'new_{cyclone_name}_test.mat')
    test = sio.loadmat(test_data)
    test = test['cyclones_test']
    test_final = test[0]

    track_ids,lengths = [],[]
    for i in range(len(train_final)):
        data_tmp = train_final[i]
        table = np.asmatrix(data_tmp)
        (m, n) = np.shape(table)
        track_ids.append(i+1)
        lengths.append(m)

    for i in range(len(test_final)):
        data_tmp = test_final[i]
        table = np.asmatrix(data_tmp)
        (m, n) = np.shape(table)
        track_ids.append(i+1)
        lengths.append(m)   

    return lengths



plt.rc('font', size = 20)

lengths = []
south_pacific_hurricane = get_number_of_cyclones('south_pacific_hurricane')
lengths.extend(south_pacific_hurricane)
south_pacific_hurricane_length = len(south_pacific_hurricane)
print(f'north indian ocean ===> {south_pacific_hurricane_length}')

p = lambda x: [element*2 for element in x]
lengths = p(lengths)


lengths = np.array(lengths)
fig, ax = plt.subplots(figsize =(10, 7))
xticks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
bins = [0,10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
xticks = p(xticks)
bins = p(bins)
ax.hist(lengths, bins = bins, color = 'lightsteelblue')
ax.set_xticks(xticks)
plt.xlabel("Cyclone Duration (in hours)")
plt.ylabel('Frequency')


#fig.set_size_inches(18.5, 10.5, forward=True)
fig.set_dpi(100)
savefile = os.path.join(os.getcwd(), 'plots', 'cyclone_lengths_histogram_south_pacific_hurricane'+ '.jpg')
plt.savefig(savefile)
#plt.show()