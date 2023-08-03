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
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=1):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, 
                            hidden_size, 
                            num_layers, 
                            batch_first=True, 
                            bidirectional=False)
        self.lin = nn.Linear(hidden_size, 
                            output_size)
        # self.act = GatedActivation()
        
        # self.bn = nn.BatchNorm1d(hidden_size)
        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, p=None):
        # x has shape [batch, channel, seq]
        x = x.permute(0, 2, 1) # input shape for LSTM with batch_first: [batch, seq, channel]

        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).requires_grad_().to(x.device)
        c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).requires_grad_().to(x.device)

        output, (hidden, cell) = self.lstm(x, (h0, c0))
<<<<<<< HEAD
        linear_output = self.lin(output) # [seq, batch, hidden_size]
        out = torch.tanh(linear_output)  
=======

        lin_output = self.lin(output) # [seq, batch, hidden_size]
>>>>>>> dc7d7e457058d97e68971c213e9cad7fdd6b23f6
        
        out = lin_output
        # out = self.bn(activated_output)
        # out = self.dropout(act_output)

        # out = out * hidden # [num_layers, batch, hidden_size]
        out = out.permute(0, 2, 1) # output shape for loss: [batch, channel, seq]
        
        return out
    

class LSTMskip(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_size=32, skip=1, num_layers=1, bias=True):
        super(LSTMskip, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_size = hidden_sizesaz
        self.conv1d = nn.Conv1d(input_size, hidden_size, kernel_size=5, padding=2)
        self.lstm = nn.LSTM(hidden_size, 
                            hidden_size,
                            num_layers, 
                            batch_first=True)
        
        self.skip = skip
        self.lin = nn.Linear(hidden_size, output_size)
        self.res = nn.Linear(skip, output_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.act1 = nn.Sigmoid()
        self.act2 = nn.Tanh()
        
    def forward(self, x, p=None):
        x = self.conv1d(x)
        x = x.permute(2,0,1) # input shape for LSTM: [seq, batch, channel]
        # x = x.permute(0,2,1) # input shape for LSTM: [batch, channel, seq]

        h0 = Variable(torch.zeros(self.num_layers, x.shape[0], self.hidden_size).requires_grad_().to(x.device))
        c0 = Variable(torch.zeros(self.num_layers, x.shape[0], self.hidden_size).requires_grad_().to(x.device))

        if self.skip:
            res = x[:,:,0:self.skip]
            output, (hidden, cell) = self.lstm(x, (h0, c0))
            output = output.permute(1, 2, 0)  # Changing to [batch, hidden_size, seq]
            output = self.bn(output)  # Apply batch normalization to the output of LSTM
            output = output.permute(2, 0, 1)  # Changing back to [seq, batch, hidden_size]
            lin_output = self.lin(output)
            lin_act = self.act2(lin_output)
            residual = self.res(res)
            lin_res = self.act1(residual)
            out = lin_output + residual
            
        else:         
            output, (hidden, cell) = self.lstm(x, (h0, c0))
            out = self.lin(output) # [seq, batch, hidden_size]
        
        out = out.permute(1,2,0) 
        return out

"""
class LSTM(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_size=32, num_layers=1, skip=1, bias_fl=True):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, 
                            hidden_size, 
                            num_layers, 
                            batch_first=True)
        
        self.lin = nn.Linear(hidden_size, output_size)
        self.bias_fl = bias_fl
        self.skip = skip
        self.hidden = None

    def forward(self, x, p=None):
        x = x.permute(2,0,1) 
        res = x[:,:,0:self.skip]

        out, self.hiden = self.lstm(x, (self.hidden))

        out = self.lin(out) + res
        
        out = out.permute(1,2,0) 
        return out
"""