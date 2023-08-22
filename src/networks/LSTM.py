import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1, dropout_prob=0.5):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, 
                            hidden_size, 
                            num_layers, 
                            batch_first=True, 
                            bidirectional=False, 
                            dropout=dropout_prob if num_layers > 1 else 0)  # Dropout only effective when num_layers > 1
        
        self.lin = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.act = nn.Tanh()

    def forward(self, x):
        x = x.permute(0, 2, 1) 

        h0 = x.new_zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = x.new_zeros(self.num_layers, x.size(0), self.hidden_size)

        output, (hidden, cell) = self.lstm(x, (h0, c0))
        output = self.dropout(output) 
        
        lin_output = self.lin(output)
        lin_output = self.act(lin_output)

        out = lin_output.permute(0, 2, 1)
        
        return out

class LSTMskip(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_size=32, skip=1, num_layers=1, bias=True):
        super(LSTMskip, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        self.conv1d = nn.Conv1d(input_size, hidden_size, kernel_size=5, padding=2)
        self.lstm = nn.LSTM(hidden_size, 
                            hidden_size,
                            num_layers, 
                            batch_first=True)

        self.skip = skip
        if self.skip:
            self.res = nn.Linear(skip, output_size)
        
        self.lin = nn.Linear(hidden_size, output_size)
        self.ln = nn.LayerNorm(output_size)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.conv1d(x)
        x = x.permute(0, 2, 1)

        h0 = x.new_zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = x.new_zeros(self.num_layers, x.size(0), self.hidden_size)

        output, (hidden, cell) = self.lstm(x, (h0, c0))
        lin_output = self.lin(output)
        lin_output = self.ln(lin_output)
        lin_output = self.act(lin_output)
        
        if self.skip:
            res = x[:, :, 0:self.skip]
            residual = self.res(res)
            lin_output = lin_output + residual

        out = lin_output.permute(0, 2, 1)
        
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