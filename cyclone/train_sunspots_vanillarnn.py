import torch
#from dataloader import trainloader,testloader
#from VanillaRNN import VanillaRNN
from south_indian_ocean_dataloader import trainloader
from VanillaRNN_south_indian_ocean import VanillaRNN
import torch.nn as nn
import numpy as np   
import matplotlib.pyplot as plt
from torchsummary import summary
import csv
import math

results_file_path = r'D:\python\BBB_RNN\data\ACFinance\RNN_results_10step.csv'

with open(results_file_path, 'w') as f:
    writer = csv.writer(f)
    header = ['rmse_train_loss', 'rmse_test_loss']
    writer.writerow(header)
    num_records = 1
    for k in range(num_records):
        input_dim = 1
        hidden_dim = 5
        output_dim = 10

        model = VanillaRNN(input_dim, hidden_dim, output_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        Loss = nn.MSELoss()
        epochs = 30
        train_mse_losses=[]
        test_mse_losses = []
        training_seq, outputs = [],[]
        for i in range(1,epochs+1):
            single_loss=0
            MSE_loss = 0

            model.train()
            for seq,labels in trainloader:
                model.zero_grad()
                Outputs = model(seq)
                loss = Loss(Outputs, labels)
                loss.backward(retain_graph = True)
                optimizer.step()
                single_loss = loss
                if i==epochs: 
                    training_seq=labels.tolist()
                    outputs = Outputs.tolist()
            train_mse_losses.append(single_loss) 
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}') 
        test_loss=0
        '''
        for seq,labels in testloader:
            outputs = model(seq)
            loss = Loss(outputs, labels)
            test_loss = loss
        '''
        rmse_train_loss = math.sqrt(train_mse_losses[-1])
        #rmse_test_loss = math.sqrt(test_loss)
        print(rmse_train_loss)
        #record = [rmse_train_loss, rmse_test_loss]
        #writer.writerow(record)
        
        
        saveplot = r'D:\python\BBB_RNN\results\track_pred_vanillarnn.png'

        plt.plot( [item[0] for item in training_seq[0]], [item[1] for item in training_seq[0]], label = 'actual')
        plt.plot([item[0] for item in outputs[0]], [item[1] for item in outputs[0]], label = 'mean')
        plt.legend();
        plt.savefig(saveplot);
        plt.show();    
        
