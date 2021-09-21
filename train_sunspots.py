import torch
from dataloader import trainloader
from BayesianRNN import BayesianRNN
import numpy as np   
import matplotlib.pyplot as plt
from torchsummary import summary

input_dim = 1
hidden_dim = 5
output_dim = 5

model = BayesianRNN(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
epochs = 20
losses = []
for i in range(1,epochs+1):
    single_loss=0
    for seq,labels in trainloader:
        loss = model.sampling_loss(seq, labels)
        loss.backward(retain_graph = True)
        optimizer.step()
        single_loss = loss
    losses.append(single_loss)
    print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
    

saveplot = r'D:\python\BBB_RNN\abs_plot.png'

plt.plot(losses, label = 'Training MSE losses')
plt.legend()
plt.savefig(saveplot)
