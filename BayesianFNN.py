import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Gaussian_weights, ScaledMixedGaussian, Gaussian
import math
import numpy as np


PI = 0.5
SIGMA1 = 5 #torch.FloatTensor([math.exp(-0)])
SIGMA2 = 5 #torch.FloatTensor([math.exp(-6)])
SAMPLES = 100

class Bayes_Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        
        #Weight
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = nn.Parameter(torch.Tensor(self.out_features, self.in_features).uniform_(-1, 1))
        self.bias_mu = nn.Parameter(torch.Tensor(self.out_features, 1).uniform_(-1, 1))
        
        self.weight_rho = nn.Parameter(torch.Tensor(self.out_features, self.in_features).uniform_(-5, -4))
        self.bias_rho  = nn.Parameter(torch.Tensor(self.out_features, 1).uniform_(-5, -4))
    
        self.weight = Gaussian_weights(self.weight_mu, self.weight_rho)
        self.bias = Gaussian_weights(self.bias_mu, self.bias_rho)     
       
        
        #Prior
        # self.prior_weight = ScaledMixedGaussian(PI, SIGMA1, SIGMA2)
        # self.prior_bias = ScaledMixedGaussian(PI, SIGMA1, SIGMA2)
        self.prior_weight = Gaussian(SIGMA1)
        self.prior_bias = Gaussian(SIGMA1)
        
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x, sampling = False, calculate_log_probs = False):
        
        if self.training or sampling:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        
        if self.training or calculate_log_probs:
            self.log_prior = self.prior_weight.log_prob(weight) + self.prior_bias.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior,self.log_variational_posterior = 0,0
        x = x.unsqueeze(2)
        Y = torch.matmul(weight, x) + bias 

        Y = Y.squeeze(2)
       
        return Y
    


class BayesianFNN(nn.Module):
    def __init__(self, dims: list, device) -> None:
        super().__init__()
        self.dims = dims
        self.layers = nn.ModuleList([Bayes_Linear(dims[ind], dims[ind+1]).to(device) for ind in range(len(self.dims)-1)])
        self.tausq = torch.ones(1)*0.00039 #nn.Parameter(torch.ones(1)*0.025)

    def forward(self, x: torch.Tensor, sampling: bool=False) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, sampling)
        return torch.sigmoid(x)

    def log_prior(self):
        return sum([layer.log_prior for layer in self.layers])
    
    def log_variational_posterior(self):
        return sum([layer.log_variational_posterior for layer in self.layers])

    def log_likelihood(self, targets, outputs):
        loss = -0.5 * torch.log(2 * torch.pi * self.tausq) - 0.5 * torch.square(targets - outputs) / self.tausq
        return torch.mean(loss)

    def sampling_loss(self, input, target, samples = SAMPLES):
        batch_size, sequence_size = list(input.size())[0], list(input.size())[1]
        Outputs = torch.zeros(samples, batch_size, self.dims[-1])
        log_priors = torch.zeros(samples)
        log_variational_posterior = torch.zeros(samples)
        # MSE_losses = torch.zeros(samples)
        

        Loss = nn.MSELoss()
        for i in range(samples):
            Outputs[i] = self(input, sampling = True)
            log_priors[i] = self.log_prior()
            log_variational_posterior[i] = self.log_variational_posterior()
            # MSE_losses[i] = Loss(Outputs[i], target)
        
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posterior.mean()
        
        log_likelihood = self.log_likelihood(targets=target, outputs=Outputs.mean(0))
        MSE_loss = Loss(target, Outputs.mean(0))

        loss = (log_variational_posterior - log_prior)/samples - log_likelihood
        
        return loss, MSE_loss, Outputs


if __name__ == '__main__':

    from dataloader import trainloader

    print(f"Is GPU available? {torch.cuda.is_available()}")
    device ="cpu" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dims = [4, 5, 1]
    fnn = BayesianFNN(dims=dims, device=device)

    for inp, out in trainloader:
        inp, out = inp.to(device), out.to(device)
        print(fnn.sampling_loss(inp, out, 37)[0])
        break
    

    optimizer = torch.optim.Adam(fnn.parameters(), lr=0.003)


    # 
    # print(fnn(inp).size())
    # print()