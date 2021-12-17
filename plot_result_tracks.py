
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from pandas.io import parsers


# -------------------------------------------

plt.style.use('bmh')

# -------------------------------------------
# ARGUMENT PARSER
# -------------------------------------------
parser = argparse.ArgumentParser(
    description="Script to plot the results for cyclone track prediction"
)

parser.add_argument(
    '-i', '--input', type=str, 
    default="/projects/Bayesian-RNN-VI/results/north-westpacificocean_test_BRNN.csv",
    help="Path to Input csv file"
)
parser.add_argument(
    '-o', '--output', type=str,
    default="output.jpg",
    help="Path to plot file"
)
parser.add_argument(
    '--start-idx', type=int, default=0,
    help="Start index of the dataframe"
)
parser.add_argument(
    '--end-idx', type=int, default=None,
    help="End index of the dataframe"
)

args = parser.parse_args()

# -------------------------------------------
# LOAD DATA
# -------------------------------------------

df = pd.read_csv(args.input, index_col=False)
df_filter = df.loc[args.start_idx: args.end_idx]


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

plt.legend(fontsize=14)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)


plt.xlabel('Longitude', fontsize=16)
plt.ylabel('Latitude', fontsize=16)

plt.savefig(args.output, dpi=300)

# -------------------------------------------



