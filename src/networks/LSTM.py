import torch
import torch.nn as nn
from torch.autograd import Variable

class GatedActivation(nn.Module):
    def __init__(self):
        super(GatedActivation, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(x) * self.sigmoid(x)


class LSTM(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_size=64, num_layers=1):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, 
                            hidden_size, 
                            num_layers, 
                            batch_first=True, 
                            bidirectional=False)
        self.lin = nn.Linear(hidden_size, 
                            output_size)

        # self.act = GatedActivation()
        self.act = nn.Tanh()

    def forward(self, x, p=None):
        x = x.permute(2,0,1) # input shape for LSTM: [seq, batch, channel]

        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).requires_grad_().to(x.device)
        c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).requires_grad_().to(x.device)

        output, (hidden, cell) = self.lstm(x, (h0, c0))
        linear_output = self.lin(output) # [seq, batch, hidden_size]
        
        out = self.act(linear_output)  
        out = out.permute(1, 2, 0) # output shape for loss: [batch, channel, seq]
        
        return out
    

class LSTMskip(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_size=32, skip=1, num_layers=1, bias=True):
        super(LSTMskip, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, 
                            hidden_size,
                            num_layers, 
                            batch_first=True)
        
        self.lin = nn.Linear(hidden_size, output_size)
        self.skip = skip
        
    def forward(self, x, p=None):
        x = x.permute(2,0,1) # input shape for LSTM: [seq, batch, channel]

        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).requires_grad_().to(x.device)
        c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).requires_grad_().to(x.device)

        if self.skip:
            res = x[:,:,0:self.skip]
            output, (hidden, cell) = self.lstm(x, (h0, c0))
            out = self.lin(output) + res
            # out = torch.cat((res, out), dim=2)
        else:         
            output, (hidden, cell) = self.lstm(x, (h0, c0))
            out = self.lin(output) # [seq, batch, hidden_size]
        
        out = out.permute(1,2,0) 
        return out
        