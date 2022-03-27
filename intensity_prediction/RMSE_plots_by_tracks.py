
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from pandas.io import parsers
import os
import numpy as np


Track_nos = 10
def get_rmse_list(df):
    track_ids = df['track_id'].unique()
    rmse_vals=[]
    for i in track_ids:
        predictions = df['prediction_speed' ].loc[df.track_id == i].to_numpy()
        targets= df['target_speed' ].loc[df.track_id == i].to_numpy()
        rmse_vals.append(np.sqrt(np.mean((predictions-targets)**2)))
        if i==Track_nos:
            break
    rmse_vals = np.array(rmse_vals)
    return rmse_vals
        
        
    
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




args = parser.parse_args()
save_file = args.output + 'RMSE_vals' + '.jpg'

rnn_df = pd.read_csv(args.input_RNN, index_col=False)


brnn_df = pd.read_csv(args.input_BRNN, index_col=False)


lstm_df = pd.read_csv(args.input_LSTM, index_col=False)


blstm_df = pd.read_csv(args.input_BLSTM, index_col=False)

track_ids = rnn_df['track_id'].unique()
#nbins = track_ids.shape[0]
nbins = Track_nos
rnn_rmse = get_rmse_list(rnn_df)
brnn_rmse = get_rmse_list(brnn_df)
lstm_rmse = get_rmse_list(lstm_df)
blstm_rmse = get_rmse_list(blstm_df)




x_axis = np.arange(nbins)

fig = plt.figure()


plt.bar(x_axis-0.2,rnn_rmse,0.1, color = 'tomato', label ='RNN')

plt.bar(x_axis-0.1,brnn_rmse,0.1, color = 'mediumseagreen', label ='BRNN')
plt.bar(x_axis,lstm_rmse,0.1, color = 'royalblue', label ='LSTM')
plt.bar(x_axis+0.1,blstm_rmse,0.1, color = 'blueviolet', label ='BLSTM')

plt.legend(fontsize=14)
#plt.title('RMSE values of each track')
plt.xlabel('Track number')
plt.ylabel('RMSE value')

plt.savefig(save_file)


