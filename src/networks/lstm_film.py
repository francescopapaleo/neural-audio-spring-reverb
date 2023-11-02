import torch
import torch.nn as nn


class FiLM(torch.nn.Module):
    def __init__(
        self,
        cond_dim,  # dim of conditioning input
        num_features,  # dim of the conv channel
        batch_norm=True,
    ):
        super().__init__()
        self.num_features = num_features
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = torch.nn.BatchNorm1d(num_features, affine=False)
        self.adaptor = torch.nn.Linear(cond_dim, num_features * 2)

    def forward(self, x, c):
        c = self.adaptor(c)
        g, b = torch.chunk(c, 2, dim=-1)
        g = g.permute(0, 2, 1)
        b = b.permute(0, 2, 1)

        if self.batch_norm:
            x = self.bn(x)  # apply BatchNorm without affine
        x = (x * g) + b  # then apply conditional affine

        return x


class LSTM_FiLM(nn.Module):
    """
    LSTM with convolutional feature extraction, optional skip connection, and FiLM layer
    """

    def __init__(
        self,
        input_size=1,
        hidden_size=32,
        num_layers=1,
        output_size=1,
        dropout_prob=0.5,
        use_skip=True,
        kernel_size=3,
        cond_dim=2,
    ):
        super(LSTM_FiLM, self).__init__()

        self.use_skip = use_skip

        # Convolutional layer and batch normalization
        self.conv1d = nn.Conv1d(
            input_size, hidden_size, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(hidden_size)

        # FiLM layer
        self.film_layer = FiLM(cond_dim=cond_dim, num_features=hidden_size)

        # LSTM layer
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0,
        )

        # Fully connected layer
        self.lin = nn.Linear(hidden_size, output_size)

        # Optional skip connection
        if self.use_skip:
            self.res = nn.Linear(input_size, output_size)

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
            res_input = x.permute(0, 2, 1)
            res = self.res(res_input)
            out = out + res

        out = out.permute(0, 2, 1)  # output shape for loss: [batch, channel, seq]

        out = out + x

        return out
