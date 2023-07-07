import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self,
                 input_size, 
                 hidden_sizes, 
                 output_size, 
                 n_layers):
        
        super(LSTMModel, self).__init__()
        self.n_layers = n_layers
        self.hidden_sizes = hidden_sizes

        self.lstm_layers = []
        for i in range(n_layers):
            if i == 0:
                self.lstm_layers.append(nn.LSTM(input_size, hidden_sizes[i], batch_first=True))
            else:
                self.lstm_layers.append(nn.LSTM(hidden_sizes[i-1], hidden_sizes[i], batch_first=True))

        self.lstm_layers = nn.ModuleList(self.lstm_layers)
        self.fc = nn.Linear(hidden_sizes[-1], output_size)
        self.act = nn.Tanh()  # activation function

    def forward(self, x):
        # Initial hidden and cell states
        h0 = [torch.zeros(x.size(0), size).to(x.device) for size in self.hidden_sizes]
        c0 = [torch.zeros(x.size(0), size).to(x.device) for size in self.hidden_sizes]

        for i in range(self.n_layers):
            # Forward propagate LSTM
            out, (h0[i], c0[i]) = self.lstm_layers[i](x, (h0[i], c0[i]))
            # Pass the output of this layer as input to next layer
            x = out

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        out = self.act(out)  # apply activation function
        return out
