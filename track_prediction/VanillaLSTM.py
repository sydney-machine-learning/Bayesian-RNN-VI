import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VanillaLSTMlinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(nn.init.kaiming_normal_(torch.empty(self.in_features, self.out_features)))
        self.bias = nn.Parameter(nn.init.kaiming_normal_(torch.empty(1, self.out_features)))
    
    def forward(self, input):
        Y = torch.matmul(input, self.weight) +self.bias
        return Y
        
class VanillaLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input_gate = VanillaLSTMlinear(self.input_dim + self.hidden_dim, self.hidden_dim)
        self.forget_gate = VanillaLSTMlinear(self.input_dim + self.hidden_dim, self.hidden_dim)
        self.cell_gate = VanillaLSTMlinear(self.input_dim + self.hidden_dim, self.hidden_dim)
        self.output_gate = VanillaLSTMlinear(self.input_dim + self.hidden_dim,self.hidden_dim)
        self.output_final = VanillaLSTMlinear(self.hidden_dim, self.output_dim)
        
        
    def forward(self, x):
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

        output = self.output_final(H_t).swapaxes(1,2)
        return output
      