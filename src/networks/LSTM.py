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
    def __init__(self, input_size=1, output_size=1, hidden_size=32, num_layers=2):
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

        self.act = GatedActivation()

    def forward(self, x, p=None):
        x = x.permute(2,0,1) # input shape for LSTM: [seq, batch, channel]

        # h0 = Variable(torch.zeros(self.num_layers, x.shape[0], self.hidden_size).requires_grad_().to(x.device))
        # c0 = Variable(torch.zeros(self.num_layers, x.shape[0], self.hidden_size).requires_grad_().to(x.device))

        out, (hidden, cell) = self.lstm(x)
        out = self.lin(out) # [num_layers, batch, hidden_size]
        
        out = self.act(out)
        # print(out.shape)

        # out = out * hidden # [num_layers, batch, hidden_size]
        out = out.permute(1, 2, 0) # output shape for loss: [batch, channel, seq]
        
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
        self.lin = nn.Linear(2 * hidden_size, output_size)
        self.act = GatedActivation()

    def forward(self, x, p=None):
        x = x.permute(2,0,1) 

        # multiply num_layers by 2 because the LSTM is bidirectional
        h0 = Variable(torch.rand(self.num_layers * 2, x.shape[0], self.hidden_size).requires_grad_().to(x.device))
        c0 = Variable(torch.rand(self.num_layers * 2, x.shape[0], self.hidden_size).requires_grad_().to(x.device))

        out, (hn, cn) = self.lstm(x, (h0, c0))

        out = self.lin(out)
        out = self.act(out)
        out = out.permute(1,2,0) 
        
        return out
    
"""
class GRU(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_size=32, num_layers=1):
        super(GRU, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, 
                            hidden_size,
                            num_layers, 
                            batch_first=True)
        
        self.lin = nn.Linear(hidden_size, output_size)
        self.hidden_lin = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, p=None):
        x = x.permute(2,0,1) # input shape for LSTM: [seq, batch, channel]

        out, (hidden, cell) = self.lstm(x)
        
        out = self.lin(out) # [num_layers, batch, hidden_size]

        # hidden = hidden.permute(1, 2, 0) # [seq, hidden_size, num_layers]

        # hidden = hidden.contiguous().(hidden.size(0))  # flatten the last two dimensions
        hidden = self.hidden_lin(hidden)
        hidden = self.hidden_lin(hidden)
        hidden_tanh = self.tanh(hidden)
        print(hidden_tanh.shape)

        out_sigmoid = self.sigmoid(out)
        # hidden_tanh = self.tanh(hidden)
        # print(out_sigmoid.shape, hidden_tanh.shape)

        out = out_sigmoid * hidden_tanh # Element-wise multiplication

        out = out.permute(1, 2, 0) # output shape for loss: [batch, channel, seq]
        
        return out
"""