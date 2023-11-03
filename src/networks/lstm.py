import torch
import torch.nn as nn

from src.networks.custom_layers import Conv1dCausal, FiLM

class LSTM(nn.Module):
    """
    LSTM with convolutional feature extraction, optional skip connection, and FiLM layer
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_layers: int,
        output_size: int,
        dropout_prob: float,
        use_skip: bool,
        kernel_size: int,
        cond_dim: int,
    ):
        super().__init__()

        self.use_skip = use_skip

        # Convolutional layer and batch normalization
        self.conv1d = Conv1dCausal(
            input_size,
            hidden_size,
            kernel_size=kernel_size,
            stride=1,
            dilation=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(hidden_size)

        # FiLM layer
        self.film_layer = FiLM(cond_dim=cond_dim, n_features=hidden_size)

        # LSTM layer
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_prob if n_layers > 1 else 0,
        )

        # Fully connected layer
        self.lin = nn.Linear(hidden_size, output_size)

        # Optional skip connection
        if self.use_skip:
            self.res = nn.Linear(hidden_size, output_size)

    def forward(self, x, c):
        # Feature extraction
        x_features = self.conv1d(x)
        x_features = self.bn1(x_features)

        # Apply FiLM modulation
        x_features = self.film_layer(x_features, c)

        # Permute for LSTM: batch, seq_len, channels
        x_permuted = x_features.permute(0, 2, 1)

        # LSTM layer
        lstm_out, _ = self.lstm(x_permuted)

        # Linear layer
        out = self.lin(lstm_out)

        # Add skip connection if it's enabled
        if self.use_skip:
            # res_input = x.permute(0, 2, 1)
            # res = self.res(res_input)
            res = self.res(x_permuted)
            out = out + res

        out = out.permute(0, 2, 1)  # output shape for loss: [batch, channel, seq]

        # out = out + x

        return out
