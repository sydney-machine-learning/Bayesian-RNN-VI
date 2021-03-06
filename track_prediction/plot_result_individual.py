
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from pandas.io import parsers
import os
import numpy as np
# -------------------------------------------

plt.style.use('bmh')

# -------------------------------------------
# ARGUMENT PARSER
# -------------------------------------------
parser = argparse.ArgumentParser(
    description="Script to plot the results for cyclone track prediction"
)

parser.add_argument(
    '-i_b', '--input_BRNN', type=str, 
    default=os.path.join(os.getcwd(), 'results', "south_pacific_hurricane_results_brnn.csv"),
    help="Path to Input csv file"
)

parser.add_argument(
    '-i_r', '--input_RNN', type=str, 
    default=os.path.join(os.getcwd(), 'results', "south_pacific_hurricane_results_rnn.csv"),
    help="Path to Input csv file" 
)

parser.add_argument(
    '-i_bl', '--input_BLSTM', type=str, 
    default=os.path.join(os.getcwd(), 'results', "south_pacific_hurricane_results_blstm.csv"),
    help="Path to Input csv file" 
)

parser.add_argument(
    '-i_l', '--input_LSTM', type=str, 
    default=os.path.join(os.getcwd(), 'results', "south_pacific_hurricane_results_lstm.csv"),
    help="Path to Input csv file" 
)

parser.add_argument(
    '-o', '--output', type=str,
    default= os.path.join(os.getcwd(), 'plots', 'south_pacific_hurricane_'),
    help="Path to plot file"
)

parser.add_argument(
    '--track_id', type=int, default=1,
    help="Track id of the cyclone"
)


args = parser.parse_args()

# -------------------------------------------
# LOAD DATA
# -------------------------------------------

df = pd.read_csv(args.input_BLSTM, index_col=False)
track_ids = df['track_id'].unique()
track_id = track_ids[args.track_id]
df_filter = df.loc[df.track_id==track_id]

save_file = args.output + 'BLSTM_track_' + str(track_id) + '.jpg'



# Target
target_longitude = df_filter.target_longitude.values
target_latitude = df_filter.target_latitude.values


# Predictions - Mean
prediction_longitude = df_filter.prediction_longitude.values
prediction_latitude = df_filter.prediction_latitude.values


# Preditions - 5 prec
prediction_5_perc_longitude = df_filter['5_percentile_longitude'].values
prediction_5_perc_latitude = df_filter['5_percentile_latitude'].values


# Preditions - 95 prec
prediction_95_perc_longitude = df_filter['95_percentile_longitude'].values
prediction_95_perc_latitude = df_filter['95_percentile_latitude'].values




# -------------------------------------------
# PLOT
# -------------------------------------------


fig = plt.figure(figsize=(14, 10))
plt.plot(target_longitude, target_latitude, label='Target', linewidth=2, alpha=0.8)
plt.plot(prediction_longitude, prediction_latitude, label='Pred-Mean', linewidth=2, alpha=0.8)
plt.plot(prediction_5_perc_longitude, prediction_5_perc_latitude,  '--', color='C5', label='Pred-5%', linewidth=1, alpha=0.6)
plt.plot(prediction_95_perc_longitude, prediction_95_perc_latitude, '--', color='C5', label='Pred-95%', linewidth=1, alpha=0.6)

plt.fill(np.append(prediction_5_perc_longitude, prediction_95_perc_longitude[::-1]), np.append(prediction_5_perc_latitude, prediction_95_perc_latitude[::-1]), 'lightgrey')
plt.legend(fontsize=22)

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)


plt.xlabel('Longitude', fontsize=24)
plt.ylabel('Latitude', fontsize=24)
plt.title('Track prediction - south pacific ocean BLSTM', fontsize=24)
plt.savefig(save_file, dpi=300)

# -------------------------------------------



