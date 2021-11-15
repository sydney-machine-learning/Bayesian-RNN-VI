import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

train_data_path = r'cyclone/data/southindianocean.csv'

IN_STEPS = 5
OUT_STEPS = 5

# Load data
df = pd.read_csv(train_data_path, index_col=False).drop(columns=['Unnamed: 0'])

class CycloneDataset(Dataset):
    def __init__(self, df, in_steps=IN_STEPS, out_steps=OUT_STEPS):
        self.track_ids = df.track_id.unique()
        self.in_steps, self.out_steps = in_steps, out_steps
        self.in_seqs, self.out_seqs = [], []
        
        for track_id in self.track_ids:
            X = df.loc[df.track_id==track_id, ['longitude', 'latitude']]
            in_seq, out_seq = self.create_sequences(X)
            if len(in_seq) == 0:
                continue
            assert len(in_seq) == len(out_seq), "Number of inputs must match the number of outputs!"
            self.in_seqs.append(in_seq)
            self.out_seqs.append(out_seq)
        
        self.in_seqs = torch.cat(self.in_seqs, dim=0)
        self.out_seqs = torch.cat(self.out_seqs, dim=0)

    def create_sequences(self, X):
        Xs, ys = [], []
        for i in range(len(X) - (self.in_steps + self.out_steps)):
            Xs.append(X.iloc[i: (i + self.in_steps)].values)
            ys.append(X.iloc[(i + self.in_steps): (i + self.in_steps + self.out_steps)].values)
        return torch.Tensor(Xs), torch.Tensor(ys)
  
    def __len__(self):
        return len(self.in_seqs)
  
    def __getitem__(self, index):
        return self.in_seqs[index], self.out_seqs[index]

dataset = CycloneDataset(df)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False) 
