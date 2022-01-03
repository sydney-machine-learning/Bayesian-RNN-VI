import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VanillaRNNlinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features).uniform_(-0.2, 0.2))
        self.bias = nn.Parameter(torch.Tensor(self.out_features, 1).uniform_(-0.2, 0.2))
    
    def forward(self, input):
        Y = torch.matmul(self.weight, input) + self.bias
        return Y
        
class VanillaRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.i_to_h = VanillaRNNlinear(input_dim + hidden_dim, hidden_dim)
        self.i_to_o = VanillaRNNlinear(input_dim + hidden_dim, output_dim)
        
    def forward(self, x):
        batch_size, sequence_size = list(x.size())[0],list(x.size())[1]
        output= torch.zeros(tuple([batch_size, self.output_dim, 1]))
        H_prev = torch.zeros(tuple([batch_size, self.hidden_dim, 1]))
        for t in range(sequence_size):
            x_t = x[:,t]
            x_t = x_t.unsqueeze(1)
            combined = torch.cat((x_t, H_prev), 1)
           
            h_t = torch.tanh(self.i_to_h(combined))
            output = self.i_to_o(combined)
           
            H_prev=h_t
        
        return output               