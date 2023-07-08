"""
@article{simionato2022,
  title = {Deep {{Learning Conditioned Modeling}} of {{Optical Compression}}},
  author = {Simionato, Riccardo and Fasciani, Stefano},
  year = {2022},
  langid = {english}
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_size=32, num_layers=2):
        super(LSTM, self).__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, 
                            hidden_size, 
                            num_layers, 
                            batch_first=True, 
                            bidirectional=False)
        self.linear = nn.Linear(hidden_size, 
                            output_size)
        self.act = nn.Tanh()

    def forward(self, x, p=None):
        x = x.permute(2,0,1) # shape for LSTM (seq, batch, channel)

        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).requires_grad_().to(x.device)
        c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).requires_grad_().to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.linear(out) # shape (num_layers, batch, hidden_size)
        out = self.act(out)

        out = out.permute(1, 2, 0) # put shape back (batch, channel, seq)
        
        return out
    

# Bidirectional LSTM

class BiLSTM(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_size=32, num_layers=2):
        super(BiLSTM, self).__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Set bidirectional to True
        self.lstm = nn.LSTM(input_size, 
                            hidden_size, 
                            num_layers, 
                            batch_first=True, 
                            bidirectional=True)
        
        # The linear layer input size is now 2 * hidden_size because the LSTM is bidirectional
        self.linear = nn.Linear(2 * hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x, p=None):
        x = x.permute(2,0,1) 

        # multiply num_layers by 2 because the LSTM is bidirectional
        h0 = torch.rand(self.num_layers * 2, x.shape[0], self.hidden_size).requires_grad_().to(x.device)  
        c0 = torch.rand(self.num_layers * 2, x.shape[0], self.hidden_size).requires_grad_().to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Concatenate the hidden states from the last layer of both directions.
        # hn = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim = 1) 

        out = self.linear(out)
        out = self.tanh(out)
        out = out.permute(1,2,0) 
        
        return out