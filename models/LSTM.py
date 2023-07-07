"""
@article{simionato2022,
  title = {Deep {{Learning Conditioned Modeling}} of {{Optical Compression}}},
  author = {Simionato, Riccardo and Fasciani, Stefano},
  year = {2022},
  langid = {english}
"""

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_size=32, num_layers=2):
        super(LSTMModel, self).__init__()

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

    def forward(self, x, c=None):
        
        # x shape (batch, channel, seq)
        x = x.permute(2,0,1) # shape for LSTM (seq, batch, channel)
                
        out, _ = self.lstm(x)
        out = torch.tanh(self.linear(out))
        out = out.permute(1,2,0) # put shape back (batch, channel, seq)
        return out


