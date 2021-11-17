import torch
#from dataloader import trainloader,testloader, trainset, testset
#from BayesianRNN import BayesianRNN
#from south_indian_ocean_dataloader import trainloader,trainsize
from BayesianRNN_south_indian_ocean import BayesianRNN
from BayesianMNN_dataloader_original import trainloader,trainset, testloader, testset
import numpy as np   
import matplotlib.pyplot as plt
from torchsummary import summary
import statistics,math,csv

input_dim = 1
hidden_dim = 5
output_dim = 1

NUM_BATCHES = len(trainloader)
#TRAIN_SIZE = trainsize
NUM_TEST_BATCHES = len(testloader)
TRAIN_SIZE = len(trainset)
TEST_SIZE = len(testset)
model = BayesianRNN(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 50
training_seq, outputs, p5, p95, se = [],[],[],[],[]
op=[]
for i in range(1,epochs+1):
    single_loss=0
    MSE_loss = 0

    model.train()
    k=0
    for seq,labels in trainloader:
        batch_losses = []
        MSE_losses = []
        batch_losses_test = []
        MSE_losses_test = []
        model.zero_grad()
        loss,MSE_loss, Outputs = model.sampling_loss(seq, labels, NUM_BATCHES)
        loss.backward(retain_graph = True)
        optimizer.step()
        single_loss = loss
        if i==epochs:
            if k:
                break
            k=1
            training_seq = torch.cat((seq, labels), 1).tolist()
            
            #training_seq=labels.tolist()
            outputs = torch.cat((seq, Outputs.mean(0)), 1).tolist()
            #outputs = Outputs.mean(0).tolist()
            p5 = torch.cat((seq,Outputs.quantile(0.05, 0)), 1).tolist()
            #p5 = Outputs.quantile(0.05, 0).tolist()
          
            p95 = torch.cat((seq, Outputs.quantile(0.95, 0)) , 1).tolist()
            #p95 = Outputs.quantile(0.95, 0).tolist()
            
        MSE_losses.append(MSE_loss*len(seq))
        batch_losses.append(loss*len(seq))
    if len(MSE_losses)>0:
        MSE_losses = torch.stack(MSE_losses)
        batch_losses = torch.stack(batch_losses)
        print(f'epoch: {i:3} loss: {batch_losses.sum()/TRAIN_SIZE:10.4f}, RMSE train: {torch.sqrt(MSE_losses.sum()/TRAIN_SIZE): .4f} ')
    '''
    for seq,labels in testloader:
        loss, MSE_loss, Outputs = model.sampling_loss(seq, labels, NUM_TEST_BATCHES)
        MSE_losses_test.append(MSE_loss*len(seq))
        batch_losses_test.append(loss*len(seq))
    

    

    MSE_losses_test = torch.stack(MSE_losses_test)
    batch_losses_test = torch.stack(batch_losses_test)        
    '''
    #print(f'epoch: {i:3} loss: {batch_losses.sum()/TRAIN_SIZE:10.4f}, RMSE train: {torch.sqrt(MSE_losses.sum()/TRAIN_SIZE): .4f} RMSE test: {torch.sqrt(MSE_losses_test.sum()/TRAIN_SIZE): .4f}')
         

train_mse, test_mse = [], []
loss_fn = torch.nn.MSELoss()
SAMPLES =100
for seq, labels in trainloader:
    out = torch.zeros(SAMPLES, len(seq), output_dim, 2)
    for i in range(SAMPLES):
        out[i] = model(seq, sampling=True)
    out = out.mean(0)
    train_mse.append(torch.pow(out - labels, 2))

# train_rmse = math.sqrt(sum(train_mse)/TRAIN_SIZE)
train_rmse = torch.sqrt(torch.mean(torch.cat(train_mse, dim=0)))
train_rmse_std = torch.std(torch.sqrt(torch.cat(train_mse, dim=0)))  

file_path =r'BBB_RNN\data\test_results.csv'
with open(file_path, 'w') as f:
    writer = csv.writer(f)
    header = ['input1_latitude', 'input1_longitude', 'input2_latitude', 'input2_longitude',
               'input3_latitude', 'input3_longitude', 'input4_latitude', 'input4_longitude',
               'target_latitude', 'target_longitude', 'prediction_latitude', 'prediction_longitude', 
               '5_percentile_latitude', '5_percentile_longitude', '95_percentile_latitude', '95_percentile_longitude',]
    writer.writerow(header)
    for seq, labels in testloader:
        out = torch.zeros(SAMPLES, len(seq), output_dim, 2)
        for i in range(SAMPLES):
           out[i] = model(seq, sampling=True)
        p5_, p95_ =out.quantile(0.05, 0).tolist(), out.quantile(0.95, 0).tolist()
        out = out.mean(0)
       
        # test_mse.append(loss_fn(out, labels)*len(seq))
        for x,y,z,a,b in zip(seq, labels, out.tolist(), p5_, p95_):
            print(z)
            record = [x[0].tolist()[0], x[0].tolist()[1], x[1].tolist()[0], x[1].tolist()[1], 
                      x[2].tolist()[0], x[2].tolist()[1], x[3].tolist()[0], x[3].tolist()[1], 
                      y.tolist()[0][0], y.tolist()[0][1], z[0][0], z[0][1], 
                      a[0][0], a[0][1], b[0][0], b[0][1]]
            writer.writerow(record)
        test_mse.append(torch.pow(out - labels, 2))

# test_rmse = math.sqrt(sum(test_mse)/TEST_SIZE)
test_rmse = torch.sqrt(torch.mean(torch.cat(test_mse, dim=0)))
test_rmse_std = torch.std(torch.sqrt(torch.cat(test_mse, dim=0)))    




print('---------------------------')
print('---------------------------')
print(f'Train RMSE ----> {train_rmse}')
print(f'Test RMSE ----> {test_rmse}')
print(f'Train RMSE std ----> {train_rmse_std}')
print(f'Test RMSE std----> {test_rmse_std}')

    



saveplot = r'D:\python\BBB_RNN\epochs_10_MSEloss.png'
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
print(type(training_seq))
print(type(training_seq[0]))
plt.plot( [item[0] for item in training_seq[0]], [item[1] for item in training_seq[0]], label = 'actual')
plt.plot([item[0] for item in outputs[0]], [item[1] for item in outputs[0]], label = 'mean')
plt.plot( [item[0] for item in p5[0]], [item[1] for item in p5[0]], label = 'p5')
plt.plot([item[0] for item in p95[0]],[item[1] for item in p95[0]], label = 'p95')
#plt.ylim(0, 1)
plt.legend();
#plt.savefig(saveplot);
plt.show();


'''
print(training_seq[0])
print(outputs[0])
print(p5[0])
print(p95[0])
'''
