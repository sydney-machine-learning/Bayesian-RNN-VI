
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from pandas.io import parsers
import os

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
    '-o', '--output', type=str,
    default= os.path.join(os.getcwd(), 'plots', 'south_pacific_hurricane_track_10.jpg'),
    help="Path to plot file"
)

parser.add_argument(
    '--track_id', type=int, default=10,
    help="Track id of the cyclone"
)


args = parser.parse_args()

# -------------------------------------------
# LOAD DATA
# -------------------------------------------

df = pd.read_csv(args.input_BRNN, index_col=False)
df_filter = df.loc[df.track_id==args.track_id]

rnn_df = pd.read_csv(args.input_RNN, index_col=False)
rnn_df_filter = rnn_df.loc[rnn_df.track_id==args.track_id]


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

#Predictions - RNN
prediction_rnn_longitude = rnn_df_filter.prediction_longitude.values
prediction_rnn_latitude = rnn_df_filter.prediction_latitude.values

# -------------------------------------------
# PLOT
# -------------------------------------------


fig = plt.figure(figsize=(14, 10))
plt.plot(target_longitude, target_latitude, label='Target', linewidth=2, alpha=0.8)
plt.plot(prediction_longitude, prediction_latitude, label='Pred-Mean', linewidth=2, alpha=0.8)
plt.plot(prediction_5_perc_longitude, prediction_5_perc_latitude,  '--', color='C5', label='Pred-5%', linewidth=1, alpha=0.6)
plt.plot(prediction_95_perc_longitude, prediction_95_perc_latitude, '--', color='C5', label='Pred-95%', linewidth=1, alpha=0.6)
plt.plot(prediction_rnn_longitude, prediction_rnn_latitude, label='Pred_rnn-', linewidth=2, alpha=0.8)

plt.legend(fontsize=14)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)


plt.xlabel('Longitude', fontsize=16)
plt.ylabel('Latitude', fontsize=16)

plt.savefig(args.output, dpi=300)

# -------------------------------------------



