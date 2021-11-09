import torch
from dataloader import trainloader,testloader
from VanillaRNN import VanillaRNN
import torch.nn as nn
import numpy as np   
import matplotlib.pyplot as plt
from torchsummary import summary
import csv
import math

results_file_path = r'D:\python\BBB_RNN\vanilla_rnn_results.csv'

with open(results_file_path, 'w') as f:
    writer = csv.writer(f)
    header = ['rmse_loss', 'train_correct', 'total_train', 'test_correct', 'total_test']
    writer.writerow(header)
    num_records = 30
    for k in range(num_records):
        input_dim = 1
        hidden_dim = 5
        output_dim = 1

        model = VanillaRNN(input_dim, hidden_dim, output_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        Loss = nn.MSELoss()
        epochs = 20
        mse_losses=[]
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
            mse_losses.append(single_loss) 
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}') 

        train_total,train_correct,test_total,test_correct = 0,0,0,0

        for seq,labels in trainloader:
            Outputs = model(seq)
            for x,y in zip(Outputs,labels):
                if x==y:
                    train_correct+=1
                train_total+=1

        for seq,labels in testloader:
            Outputs = model(seq)
            for x,y in zip(Outputs,labels):
                if x==y:
                    test_correct+=1
                test_total+=1         
        
        rmse_loss = math.sqrt(mse_losses[-1])
        
        record = [rmse_loss, train_correct, train_total, test_correct, test_total]
        writer.writerow(record)
    



'''
saveplot = r'D:\python\BBB_RNN\epochs_10_MSEloss.png'
'''
"""
plt.plot(losses, label = 'Training KL Divergence loss')
plt.legend()
plt.savefig(saveplot)
"""
"""
plt.plot(mse_losses, label = 'Training MSE loss')
plt.legend()
plt.savefig(saveplot);
"""
'''
plt.plot( training_seq[0], label = 'actual')
plt.plot(outputs[0], label = 'mean')
#plt.ylim(0, 1)
plt.legend();
#plt.savefig(saveplot);
plt.show();
'''
'''
print(training_seq[0])
print(outputs[0])
print(p5[0])
print(p95[0])
'''



#Accuracy plot

