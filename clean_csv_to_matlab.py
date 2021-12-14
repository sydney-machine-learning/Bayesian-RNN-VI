import csv
import os
import pandas as pd
import numpy as np
import scipy.io as sio

data_file_path = os.path.join(os.getcwd(),'data','south_pacific_hurricane', 'south_pacific_hurricane.csv') 
new_mat_file_path_train = os.path.join(os.getcwd(), 'data','south_pacific_hurricane', 'new_south_pacific_hurricane_train.mat')
new_mat_file_path_test = os.path.join(os.getcwd(), 'data','south_pacific_hurricane', 'new_south_pacific_hurricane_test.mat')
                    

                         
df = pd.read_csv(data_file_path)
columns = ['track_id', 'date', 'latitude', 'longitude', 'speed']
df.columns = columns
df =df.dropna()
df.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).dropna()


track_ids = df.track_id.unique()
data = []
       

curr_track=track_ids[0]
curr_list=[]
x=[]
count=0
for index,row in df.iterrows():
    if row['track_id']=='#VALUE!' or row['longitude']=='#VALUE!' or row['latitude']=='#VALUE!' or row['speed']=='#VALUE!':
        continue
    track_id = np.double(row['track_id'])
    longitude = np.double(row['longitude'])
    latitude = np.double(row['latitude'])
    speed = np.double(row['speed'])
    #print(track_id)
    if track_id==curr_track:
        curr_list.append([track_id, latitude, longitude, speed])
    else:
        x0,y0 = curr_list[0][1],curr_list[0][2]
        for j in range(len(curr_list)):
            curr_list[j][1] -=x0
            curr_list[j][2] -=y0 
        if len(curr_list)>5:
            curr_list = np.array(curr_list, dtype =np.double)
            x.append(np.array(curr_list, dtype =np.double))
        curr_track = track_id
        curr_list=[]
        curr_list.append([track_id, latitude, longitude, speed])
        count+=1

def list_splitter(list_to_split, ratio):
    elements = len(list_to_split)
    middle = int(elements * ratio)
    return [list_to_split[:middle], list_to_split[middle:]]

train, test = list_splitter(x, 0.7)              
train,test = np.array(train), np.array(test)
print(train.shape)
print(test.shape) 
train,test = np.expand_dims(train, 1), np.expand_dims(test, 1)
train, test = np.swapaxes(train, 0, 1), np.swapaxes(test, 0, 1)

sio.savemat(new_mat_file_path_train, {'cyclones_train': train})
sio.savemat(new_mat_file_path_test, {'cyclones_test': test})

    

    
    