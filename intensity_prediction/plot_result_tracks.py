
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
    default=os.path.join(os.getcwd(), 'results', "south_indian_hurricane_results_blstm.csv"),
    help="Path to Input csv file"
)

parser.add_argument(
    '-i_r', '--input_RNN', type=str, 
    default=os.path.join(os.getcwd(), 'results', "south_indian_hurricane_results_rnn.csv"),
    help="Path to Input csv file" 
)

parser.add_argument(
    '-o', '--output', type=str,
    default= os.path.join(os.getcwd(), 'plots', 'south_indian_hurricane_track_2_blstm.jpg'),
    help="Path to plot file"
)

parser.add_argument(
    '--track_id', type=int, default=2,
    help="Track id of the cyclone"
)


args = parser.parse_args()

# -------------------------------------------
# LOAD DATA
# -------------------------------------------

df = pd.read_csv(args.input_BRNN, index_col=False)
df_filter = df.loc[df.track_id==args.track_id]





# Target
target_speed = df_filter.target_speed.values
target_speed = target_speed.astype(np.float)
target_speed = np.round(target_speed, decimals=2)

# Predictions - Mean
prediction_speed = df_filter.prediction_speed.values
prediction_speed = prediction_speed.astype(np.float)
prediction_speed = np.round(prediction_speed, decimals=2)

# Preditions - 5 prec
prediction_5_perc_speed = df_filter['5_percentile_speed'].values
prediction_5_perc_speed = prediction_5_perc_speed.astype(np.float)
prediction_5_perc_speed = np.round(prediction_5_perc_speed, decimals=2)

# Preditions - 95 prec
prediction_95_perc_speed = df_filter['95_percentile_speed'].values
prediction_95_perc_speed = prediction_95_perc_speed.astype(np.float)
prediction_95_perc_speed = np.round(prediction_95_perc_speed, decimals=2)

# -------------------------------------------
# PLOT
# -------------------------------------------


x = list(range(df_filter.shape[0]))
x_ = list(range(df_filter.shape[0]*2))
#fig = plt.figure(figsize=(14, 10))
plt.plot(x, target_speed, label='Target', linewidth=2, alpha=0.8)
plt.plot(x, prediction_speed, label='Pred-Mean', linewidth=2, alpha=0.8)
plt.plot(x, prediction_5_perc_speed,  '--', color='C5', label='Pred-5%', linewidth=1, alpha=0.6)
plt.plot(x, prediction_95_perc_speed, '--', color='C5', label='Pred-95%', linewidth=1, alpha=0.6)


plt.fill(np.append(np.array(x), np.array(x)[::-1]), np.append(prediction_5_perc_speed, prediction_95_perc_speed[::-1]), 'lightgrey')
plt.legend(fontsize=14)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)


plt.xlabel('observation no.', fontsize=16)
plt.ylabel('speed', fontsize=16)

plt.savefig(args.output, dpi=300)

# -------------------------------------------



