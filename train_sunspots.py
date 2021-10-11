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
epochs = 5
losses = []
mse_losses=[]
training_seq, outputs, p5, p95 = [],[],[],[]
op=[]
for i in range(1,epochs+1):
    single_loss=0
    MSE_loss = 0

    model.train()
    for seq,labels in trainloader:
        model.zero_grad()
        loss,MSE_loss, Outputs = model.sampling_loss(seq, labels)
        loss.backward(retain_graph = True)
        optimizer.step()
        single_loss = loss
        if i==epochs: 
            training_seq=labels.tolist()
            outputs = Outputs.mean(0).tolist()
            p5 = Outputs.quantile(0.05, 0).tolist()
            p95 = Outputs.quantile(0.95, 0).tolist()
    losses.append(single_loss) 
    mse_losses.append(MSE_loss) 
    print(f'epoch: {i:3} loss: {single_loss.item():10.8f}') 
                



saveplot = r'D:\python\BBB_RNN\MSE.png'
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

plt.plot( training_seq[0], label = 'actual')
plt.plot(outputs[0], label = 'mean')
plt.plot( p5[0], label = 'p5')
plt.plot(p95[0], label = 'p95')
plt.ylim(0, 1)
plt.legend();
plt.show();
