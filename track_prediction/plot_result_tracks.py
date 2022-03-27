
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from pandas.io import parsers
import os
import numpy as np
import seaborn as sns
import itertools
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
    default=os.path.join(os.getcwd(), 'percentileresults', "south_pacific_hurricane_results_brnn.csv"),
    help="Path to Input csv file"
)

parser.add_argument(
    '-i_r', '--input_RNN', type=str, 
    default=os.path.join(os.getcwd(), 'percentileresults', "south_pacific_hurricane_results_rnn.csv"),
    help="Path to Input csv file" 
)

parser.add_argument(
    '-i_bl', '--input_BLSTM', type=str, 
    default=os.path.join(os.getcwd(), 'percentileresults', "south_pacific_hurricane_results_blstm.csv"),
    help="Path to Input csv file" 
)

parser.add_argument(
    '-i_l', '--input_LSTM', type=str, 
    default=os.path.join(os.getcwd(), 'percentileresults', "south_pacific_hurricane_results_lstm.csv"),
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
blstm_df_filter = df.loc[df.track_id==track_id]

save_file = args.output + 'combined_predictions_' + str(track_id) +'_Paired'+ '.jpg'

rnn_df = pd.read_csv(args.input_RNN, index_col=False)
rnn_df_filter = rnn_df.loc[rnn_df.track_id==track_id]

lstm_df = pd.read_csv(args.input_LSTM, index_col=False)
lstm_df_filter = lstm_df.loc[rnn_df.track_id==track_id]

brnn_df = pd.read_csv(args.input_BRNN, index_col=False)
brnn_df_filter = brnn_df.loc[rnn_df.track_id==track_id]


offset_long,offset_lat =  164.7, 10.5
# Target
target_longitude = blstm_df_filter.target_longitude.values + offset_long
target_latitude = blstm_df_filter.target_latitude.values + offset_lat



# Predictions - BLSTM
prediction_blstm_longitude = blstm_df_filter.prediction_longitude.values + offset_long
prediction_blstm_latitude = blstm_df_filter.prediction_latitude.values + offset_lat


#Predictions - RNN
prediction_rnn_longitude = rnn_df_filter.prediction_longitude.values + offset_long
prediction_rnn_latitude = rnn_df_filter.prediction_latitude.values + offset_lat

#Predictions - LSTM
prediction_lstm_longitude = lstm_df_filter.prediction_longitude.values + offset_long
prediction_lstm_latitude = lstm_df_filter.prediction_latitude.values + offset_lat
    
#Predictions - BRNN
prediction_brnn_longitude = brnn_df_filter.prediction_longitude.values + offset_long
prediction_brnn_latitude = brnn_df_filter.prediction_latitude.values + offset_lat


# -------------------------------------------
# PLOT
# -------------------------------------------

'''
fig = plt.figure(figsize=(14, 10))


plt.plot(target_longitude, target_latitude, label='Target',color = 'gold', linewidth=2, alpha=0.8)
plt.plot(prediction_blstm_longitude, prediction_blstm_latitude, label='BLSTM',color =  'red', linewidth=2, alpha=0.8)
plt.plot(prediction_lstm_longitude, prediction_lstm_latitude, label='LSTM',color = 'lightcoral', linewidth=2, alpha=0.8)
plt.plot(prediction_brnn_longitude, prediction_brnn_latitude, label='BRNN',color = 'navy', linewidth=2, alpha=0.8)
plt.plot(prediction_rnn_longitude, prediction_rnn_latitude, label='RNN', color = 'dodgerblue' ,linewidth=2, alpha=0.8)


plt.legend(fontsize=14)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)


plt.xlabel('Longitude '+ u"\N{DEGREE SIGN}" + 'E', fontsize=16)
plt.ylabel('Latitude ' + u"\N{DEGREE SIGN}" + 'N', fontsize=16)


plt.savefig(save_file, dpi=300)
'''
# -------------------------------------------

#convert into dataframe
x_target = np.array(['target' for i in range(len(target_longitude))])
x_blstm = np.array(['BLSTM' for i in range(len(prediction_blstm_longitude))])
x_rnn = np.array(['RNN' for i in range(len(prediction_rnn_longitude))])
x_lstm = np.array(['LSTM' for i in range(len(prediction_lstm_longitude))])
x_brnn = np.array(['BRNN' for i in range(len(prediction_brnn_longitude))])
print(f'{len(target_longitude)}, {len(prediction_blstm_longitude)} {len(prediction_rnn_longitude)} {len(prediction_lstm_longitude)} {len(prediction_brnn_longitude)}')
longitudes = np.concatenate((target_longitude, prediction_blstm_longitude, prediction_rnn_longitude, prediction_lstm_longitude, prediction_brnn_longitude))
latitudes = np.concatenate((target_latitude, prediction_blstm_latitude, prediction_rnn_latitude, prediction_lstm_latitude, prediction_brnn_latitude))
categories = np.concatenate((x_target, x_blstm, x_rnn, x_lstm, x_brnn))
series_long = pd.Series(longitudes)
series_lat = pd.Series(latitudes)
series_cat = pd.Series(categories)
print(series_long)
df = pd.concat([series_long, series_lat, series_cat], axis=1)
df.columns = ['longitude', 'latitude', 'Legend']
hue_order = ['target','BLSTM', 'LSTM', 'BRNN', 'RNN']
ax = sns.lineplot(x = 'longitude', y = 'latitude', data =df, hue = 'Legend',hue_order = hue_order,style = 'Legend', sort=False, estimator = None, palette = 'Paired')
ax.set(xlabel = 'Longitude \N{DEGREE SIGN}', ylabel = 'Latitude \N{DEGREE SIGN}')
#plt.show()
plt.savefig(save_file)