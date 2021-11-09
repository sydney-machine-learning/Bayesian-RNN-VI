import argparse
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

## -----------------------------------------------------------------
## Argument parser
## -----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Create Image data from North Indian Ocean Cyclone Dataset")
parser.add_argument(
    '--src', type=str, 
    default='./data/southindianocean_jtwc.csv', 
    help="Path to the source csv file"
)
parser.add_argument(
    '--dst', type=str, 
    default='./data/North_Indian_Ocean/images/', 
    help="Path to store the processed data"
)
parser.add_argument(
    '--format', type=str, 
    default='numpy', 
    help="Save as numpy or jpg?"
)

args = parser.parse_args()


## -----------------------------------------------------------------
## 1. READ DATA FILE
## -----------------------------------------------------------------
cyclone_df = pd.read_csv(
    args.src, delimiter = ",", header = None, 
    names =['id', 'timestamp', 'longitude', 'latitude', 'speed']
)

# Format the timestamp column
cyclone_df['timestamp'] = pd.to_datetime(cyclone_df['timestamp'], format='%Y%m%d%H')

# Convert logitude to float type
cyclone_df.loc[cyclone_df.longitude=='#VALUE!', 'longitude'] = np.nan
cyclone_df.loc[cyclone_df.speed=='-999', 'speed'] = np.nan
cyclone_df.dropna(inplace=True)


cyclone_df.reset_index(drop=True, inplace=True)
cyclone_df = cyclone_df.astype({'id' : 'int', 'longitude' : 'float64', 'latitude' : 'float64'})


## -----------------------------------------------------------------
## 2. ASSIGN UNIQUE ID TO EACH TRACK
## -----------------------------------------------------------------

# Compare current and next id
curr = cyclone_df.loc[1:, 'id'].values
prev = cyclone_df.loc[:len(cyclone_df)-2, 'id'].values

# Indices where the id changes
change = np.concatenate([[True], curr != prev])
cyclone_df.loc[:, 'change'] = change

# Assign a new track_id to the row when change value is true
track_id = 0

def assign_cyclone_id(change):
    global track_id
    if change:
        track_id += 1
    return track_id

cyclone_df.loc[:, 'track_id'] = cyclone_df.change.apply(assign_cyclone_id)


cyclone_df.to_csv(path_or_buf = './southindianocean.csv')

    
