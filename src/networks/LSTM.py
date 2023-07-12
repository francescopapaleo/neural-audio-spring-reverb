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

        # self.act = GatedActivation()

    def forward(self, x, p=None):
        x = x.permute(2,0,1) # input shape for LSTM: [seq, batch, channel]

        h0 = Variable(torch.rand(self.num_layers, x.shape[0], self.hidden_size).requires_grad_().to(x.device))
        c0 = Variable(torch.rand(self.num_layers, x.shape[0], self.hidden_size).requires_grad_().to(x.device))

        output, (hidden, cell) = self.lstm(x, (h0, c0))
        linear_output = self.lin(output) # [seq, batch, hidden_size]
        out = torch.tanh(linear_output)  
        
        # out = self.act(out)
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
        # self.act = GatedActivation()

    def forward(self, x, p=None):
        x = x.permute(2,0,1) 

        # multiply num_layers by 2 because the LSTM is bidirectional
        h0 = Variable(torch.rand(self.num_layers * 2, x.shape[0], self.hidden_size).requires_grad_().to(x.device))
        c0 = Variable(torch.rand(self.num_layers * 2, x.shape[0], self.hidden_size).requires_grad_().to(x.device))

        out, (hn, cn) = self.lstm(x, (h0, c0))

        out = torch.tanh(self.lin(out))
        # out = self.act(out)
        out = out.permute(1,2,0) 
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

        h0 = Variable(torch.rand(self.num_layers, x.shape[0], self.hidden_size).requires_grad_().to(x.device))
        c0 = Variable(torch.rand(self.num_layers, x.shape[0], self.hidden_size).requires_grad_().to(x.device))

        if self.skip:
            res = x[:,:,0:self.skip]
            output, (hidden, cell) = self.lstm(x, (h0, c0))
            out = self.lin(output) + res
            # out = torch.cat((res, out), dim=2)
            # out = torch.nn.functional.softmax(out, dim=0)        
        else:         
            output, (hidden, cell) = self.lstm(x, (h0, c0))
            out = self.lin(output) # [seq, batch, hidden_size]
        
        out = out.permute(1,2,0) 
        return out
        