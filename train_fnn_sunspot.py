import torch
from dataloader import trainloader, testloader, train_dataset, test_dataset
from BayesianFNN import BayesianFNN
from BayesianRNN import BayesianRNN
import numpy as np   
import matplotlib.pyplot as plt
from torchsummary import summary
import statistics, math


dims = [4, 5, 1]

print(f"Is GPU available? {torch.cuda.is_available()}")
device ="cpu" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

NUM_BATCHES = len(trainloader)
NUM_TEST_BATCHES = len(testloader)
TRAIN_SIZE = len(train_dataset)
TEST_SIZE = len(test_dataset)

model =  BayesianFNN(dims=dims, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 200
losses = []
mse_losses=[]
training_seq, outputs, p5, p95 = [],[],[],[]
op=[]


for i in range(1,epochs+1):
    batch_losses = []
    MSE_losses = []
    batch_losses_test = []
    MSE_losses_test = []


    model.train()
    for seq, labels in trainloader:
        model.zero_grad()
        loss, MSE_loss, Outputs = model.sampling_loss(seq, labels)
        loss.backward(retain_graph = True)
        optimizer.step()
        single_loss = loss
        
        if i==epochs: 
            training_seq=labels.tolist()
            outputs = Outputs.mean(0).tolist()
            p5 = Outputs.quantile(0.05, 0).tolist()
            p95 = Outputs.quantile(0.95, 0).tolist()
        
        MSE_losses.append(MSE_loss*len(seq))
        batch_losses.append(loss*len(seq))
    
    for seq, labels in testloader:
        loss, MSE_loss, Outputs = model.sampling_loss(seq, labels)
        MSE_losses_test.append(MSE_loss*len(seq))
        batch_losses_test.append(loss*len(seq))
    
    MSE_losses = torch.stack(MSE_losses)
    batch_losses = torch.stack(batch_losses)

    MSE_losses_test = torch.stack(MSE_losses_test)
    batch_losses_test = torch.stack(batch_losses_test)
    
    print(f'epoch: {i:3} loss: {batch_losses.sum()/TRAIN_SIZE:10.4f}, RMSE train: {torch.sqrt(MSE_losses.sum()/TRAIN_SIZE): .4f} RMSE test: {torch.sqrt(MSE_losses_test.sum()/TRAIN_SIZE): .4f}') 


train_mse, test_mse = [], []
loss_fn = torch.nn.MSELoss()

for seq, labels in trainloader:
    out = model(seq, sampling=False)
    train_mse.append(torch.pow(out - labels, 2))

# train_rmse = math.sqrt(sum(train_mse)/TRAIN_SIZE)
train_rmse = torch.sqrt(torch.mean(torch.cat(train_mse, dim=0)))

for seq, labels in testloader:
    out = model(seq, sampling=False)
    # test_mse.append(loss_fn(out, labels)*len(seq))
    train_mse.append(torch.pow(out - labels, 2))

# test_rmse = math.sqrt(sum(test_mse)/TEST_SIZE)
test_rmse = torch.sqrt(torch.mean(torch.cat(train_mse, dim=0)))
    




print('---------------------------')
print('---------------------------')
print(f'Train RMSE ----> {train_rmse}')
print(f'Test RMSE ----> {test_rmse}')
# print(f'test average ----> {test_avg}')
# print(f'test_stddev ----> {test_stddev}')

    



# saveplot = r'D:\python\BBB_RNN\epochs_10_MSEloss.png'