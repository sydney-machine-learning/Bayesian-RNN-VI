import torch
from dataloader import trainloader,testloader
from BayesianRNN import BayesianRNN
import numpy as np   
import matplotlib.pyplot as plt
from torchsummary import summary
import statistics,math

input_dim = 1
hidden_dim = 5
output_dim = 10

NUM_BATCHES = len(trainloader)
NUM_TEST_BATCHES = len(testloader)
model = BayesianRNN(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
epochs = 20
losses = []
mse_losses=[]
training_seq, outputs, p5, p95 = [],[],[],[]
op=[]
for i in range(1,epochs+1):
    single_loss=0
    MSE_loss = 0

    model.train()
    for seq, labels in trainloader:
        model.zero_grad()
        loss,MSE_loss, Outputs = model.sampling_loss(seq, labels, NUM_BATCHES)
        loss.backward(retain_graph = True)
        optimizer.step()
        single_loss = loss
        if i==epochs: 
            training_seq=labels.tolist()
            outputs = Outputs.mean(0).tolist()
            p5 = Outputs.quantile(0.05, 0).tolist()
            p95 = Outputs.quantile(0.95, 0).tolist()
            mse_losses = MSE_loss
    losses.append(single_loss) 
    print(f'epoch: {i:3} loss: {single_loss.item():10.8f}') 

test_mse_losses=[]
for seq,labels in testloader:
    test_mse_losses = model.testing(seq, labels, NUM_TEST_BATCHES)

train_losses,test_losses=[],[]
for loss in mse_losses:
    train_losses.append(math.sqrt(loss.item()))
for loss in test_mse_losses:
    test_losses.append(math.sqrt(loss.item()))
train_avg = statistics.mean(train_losses)
train_stddev = statistics.stdev(train_losses)
test_avg = statistics.mean(test_losses)
test_stddev = statistics.stdev(test_losses)

print('---------------------------')
print('---------------------------')
print(f'Train_Average ----> {train_avg}')
print(f'train_stddev ----> {train_stddev}')
print(f'test average ----> {test_avg}')
print(f'test_stddev ----> {test_stddev}')

saveplot = r'MSE.png'
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
"""
plt.plot( training_seq[0], label = 'actual')
plt.plot(outputs[0], label = 'mean')
plt.plot( p5[0], label = 'p5')
plt.plot(p95[0], label = 'p95')
#plt.ylim(0, 1)
plt.legend();
plt.savefig(saveplot);
plt.show();
"""

'''
print(training_seq[0])
print(outputs[0])
print(p5[0])
print(p95[0])
'''
