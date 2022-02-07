import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import Gaussian_weights, ScaledMixedGaussian
import math



PI = 0.5
SIGMA1 = 6
SIGMA2 = 6
SAMPLES = 100

class BayesLSTM_Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        
        #Weight
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = nn.Parameter(torch.Tensor(self.in_features, self.out_features).uniform_(-0.2, 0.2))
        self.bias_mu = nn.Parameter(torch.Tensor(1,self.out_features).uniform_(-0.2, 0.2))
        
        self.weight_rho = nn.Parameter(torch.Tensor(self.in_features, self.out_features).uniform_(-5, -4))
        self.bias_rho  = nn.Parameter(torch.Tensor(1,self.out_features).uniform_(-5, -4))
    
        self.weight = Gaussian_weights(self.weight_mu, self.weight_rho)
        self.bias = Gaussian_weights(self.bias_mu, self.bias_rho)     
       
        
        #Prior
        self.prior_weight = ScaledMixedGaussian(PI, SIGMA1, SIGMA2)
        self.prior_bias = ScaledMixedGaussian(PI, SIGMA1, SIGMA2)
        
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
        Y = torch.matmul(x, weight) + bias 

        Y = Y.squeeze(2)
       
        return Y
    
class BayesianLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input_gate = BayesLSTM_Linear(self.input_dim + self.hidden_dim, self.hidden_dim)
        self.forget_gate = BayesLSTM_Linear(self.input_dim + self.hidden_dim, self.hidden_dim)
        self.cell_gate = BayesLSTM_Linear(self.input_dim + self.hidden_dim, self.hidden_dim)
        self.output_gate = BayesLSTM_Linear(self.input_dim + self.hidden_dim,self.hidden_dim)
        self.output_final = BayesLSTM_Linear(self.hidden_dim, self.output_dim)
        
    def forward(self, x, sampling = False):
        batch_size, sequence_size = list(x.size())[0],list(x.size())[1]
        H_prev,C_prev = (nn.init.kaiming_normal_(torch.empty(tuple([batch_size, 2, self.hidden_dim]))), 
                        nn.init.kaiming_normal_(torch.empty(tuple([batch_size, 2, self.hidden_dim]))))
        for t in range(sequence_size):
            x_t = x[:,t]
            x_t = x_t.unsqueeze(-1)
            combined = torch.cat((x_t, H_prev), -1)
            I_t = torch.sigmoid(self.input_gate(combined))
            F_t = torch.sigmoid(self.forget_gate(combined) )
            C_hat_t = torch.tanh(self.cell_gate(combined))
            O_t = self.output_gate(combined)
            
            C_t = torch.mul(F_t, C_prev) + torch.mul(I_t,C_hat_t) 
           
            H_t = torch.mul(O_t, torch.tanh(C_t))
            
            H_prev = H_t 
            C_prev = C_t

        output = self.output_final(H_t).unsqueeze(1)
        return output
    
    def log_prior(self):
        return (self.input_gate.log_prior + self.forget_gate.log_prior 
                + self.cell_gate.log_prior + self.output_gate.log_prior + self.output_final.log_prior)
    
    def log_variational_posterior(self):
        return (self.input_gate.log_variational_posterior + self.forget_gate.log_variational_posterior 
                + self.cell_gate.log_variational_posterior + self.output_gate.log_variational_posterior + self.output_final.log_variational_posterior)
    
    def sampling_loss(self, input, target,NUM_BATCHES, samples = SAMPLES):
        batch_size, sequence_size = list(input.size())[0],list(input.size())[1]
        Outputs = torch.zeros(samples, batch_size, self.output_dim, 2)
        log_priors = torch.zeros(samples)
        log_variational_posterior = torch.zeros(samples)
        
        for i in range(samples):
            Outputs[i] = self(input, sampling = True)
            log_priors[i] = self.log_prior()
            log_variational_posterior[i] = self.log_variational_posterior()  
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posterior.mean()
        Loss = nn.MSELoss()
        var = torch.ones(target.size()[0], 1, requires_grad= True)
        negative_log_likelihood = Loss(Outputs.mean(0), target)
        MSE_loss =negative_log_likelihood
        loss = (log_variational_posterior - log_prior)/NUM_BATCHES + negative_log_likelihood
        
        return loss, MSE_loss, Outputs     
    
    def testing(self, input, samples = SAMPLES):
        batch_size, sequence_size = list(input.size())[0],list(input.size())[1]
        Outputs = torch.zeros(samples, batch_size, self.output_dim, 2)
        MSE_loss = []
        Loss = nn.MSELoss()
        for i in range(samples):
            Outputs[i] = self(input, sampling = True)
        return Outputs