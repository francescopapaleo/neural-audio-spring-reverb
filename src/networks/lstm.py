import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=1, 
                            batch_first=True,
                            bidirectional=False)

        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, x):   # x has shape [batch, channel, seq]
        x = x.permute(0, 2, 1)      # input shape for LSTM with batch_first: [batch, seq, channel]

        # h0 = .new_zeros(self.num_layers, x.shape[0], self.hidden_size)
        # c0 = .new_zeros(self.num_layers, x.shape[0], self.hidden_size)

        output, _ = self.lstm(x)

        lin_output = self.lin(output) # [seq, batch, hidden_size]
        
        out = lin_output

        # out = out * hidden # [num_layers, batch, hidden_size]
        out = out.permute(0, 2, 1) # output shape for loss: [batch, channel, seq]
        
        return out


class LstmConvSkip(nn.Module):
    """
    LSTM with convolutional feature extraction and optional skip connection
    """
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1, dropout_prob=0.5, use_skip=True, kernel_size=3):
        super(LstmConvSkip, self).__init__()
        
        self.use_skip = use_skip

        # Convolutional layer for feature extraction
        self.conv1d = nn.Conv1d(input_size, hidden_size, kernel_size=kernel_size, padding=kernel_size // 2)
        
        # LSTM layer
        self.lstm = nn.LSTM(hidden_size, 
                            hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=dropout_prob if num_layers > 1 else 0)
        
        # Fully connected layer
        self.lin = nn.Linear(hidden_size, output_size)
        
        # Optional skip connection
        if self.use_skip:
            self.res = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        # Feature extraction
        x_features = self.conv1d(x)

        # Permute for LSTM: batch, seq_len, channels
        x_permuted = x_features.permute(0, 2, 1)
        
        # Lstm layer
        lstm_out, _ = self.lstm(x_permuted)
        
        # Linear layer
        out = self.lin(lstm_out)
        
        # Add skip connection if it's enabled
        if self.use_skip:
            res_input = x.permute(0, 2, 1)
            res = self.res(res_input)
            out = out + res
        
        out = out.permute(0, 2, 1) # output shape for loss: [batch, channel, seq]
        
        return out


