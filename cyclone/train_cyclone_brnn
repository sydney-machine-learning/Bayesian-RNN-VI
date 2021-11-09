import torch
#from dataloader import trainloader,testloader
from south_indian_ocean_dataloader import trainloader
from BayesianRNN_south_indian_ocean import BayesianRNN
import numpy as np   
import matplotlib.pyplot as plt
from torchsummary import summary
import statistics,math

input_dim = 1
hidden_dim = 5
output_dim = 10

NUM_BATCHES = len(trainloader)
#NUM_TEST_BATCHES = len(testloader)
model = BayesianRNN(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
epochs = 30
losses = []
mse_losses=[]
training_seq, outputs, p5, p95 = [],[],[],[]
op=[]
for i in range(1,epochs+1):
    single_loss=0
    MSE_loss = 0

    model.train()
    count=0
    for seq,labels in trainloader:
        count+=1
        model.zero_grad()
        loss,MSE_loss, Outputs = model.sampling_loss(seq, labels, NUM_BATCHES)
        loss.backward(retain_graph = True)
        optimizer.step()
        single_loss = MSE_loss
        if i==epochs: 
            training_seq=labels.tolist()
            outputs = Outputs.mean(0).tolist()
            p5 = Outputs.quantile(0.05, 0).tolist()
            p95 = Outputs.quantile(0.95, 0).tolist()
            losses.append(single_loss) 
    print(f'epoch: {i:3} loss: {single_loss.item():10.8f}') 
'''
test_mse_losses=[]
for seq,labels in testloader:
    test_mse_losses.append(model.testing(seq, labels, NUM_TEST_BATCHES))
'''
train_losses,test_losses=[],[]

for loss in losses:
    train_losses.append(math.sqrt(loss.item()))
'''
for loss in test_mse_losses:
    test_losses.append(math.sqrt(loss.item()))
'''
#print(train_losses)
train_avg = statistics.mean(train_losses)
train_stddev = statistics.stdev(train_losses)
#test_avg = statistics.mean(test_losses)
#test_stddev = statistics.stdev(test_losses)

print('---------------------------')
print('---------------------------')
print(f'Train_Average ----> {train_avg}')
print(f'train_stddev ----> {train_stddev}')
#print(f'test average ----> {test_avg}')
#print(f'test_stddev ----> {test_stddev}')

    



saveplot = r'D:\python\BBB_RNN\results\track_pred_BRNN.png'
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

plt.plot( [item[0] for item in training_seq[0]], [item[1] for item in training_seq[0]], label = 'actual')
plt.plot([item[0] for item in outputs[0]], [item[1] for item in outputs[0]], label = 'mean')
plt.plot( [item[0] for item in p5[0]], [item[1] for item in p5[0]], label = 'p5')
plt.plot([item[0] for item in p95[0]],[item[1] for item in p95[0]], label = 'p95')
#plt.ylim(0, 1)
plt.legend();
plt.savefig(saveplot);
plt.show();


'''
print(training_seq[0])
print(outputs[0])
print(p5[0])
print(p95[0])
'''
