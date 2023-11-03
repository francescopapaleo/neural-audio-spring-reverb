import torch
import torch.nn as nn

from src.networks.custom_layers import Conv1dCausal, FiLM

class GRU(nn.Module):
    """
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        n_layers,
        output_size,
        dropout_prob,
        use_skip,
        kernel_size,
        cond_dim,
    ):
        super().__init__()

        assert cond_dim > 0, "cond_dim must be greater than 0 for FiLM layer."

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
        self.prelu1 = nn.PReLU()

        # FiLM layer
        self.film_layer = FiLM(cond_dim=cond_dim, n_features=hidden_size)

        # GRU layer
        self.gru = nn.GRU(
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
            self.res = nn.Linear(input_size, output_size)

    def forward(self, x, c):
        if c is None:
            raise ValueError("Conditional input 'c' is required for FiLM layer.")

        # Feature extraction
        x_features = self.conv1d(x)
        x_features = self.bn1(x_features)
        x_features = self.prelu1(x_features)

        # Apply FiLM modulation
        x_features = self.film_layer(x_features, c)

        # Permute for GRU
        x_permuted = x_features.permute(0, 2, 1)

        # GRU layer
        gru_out, _ = self.gru(x_permuted)

        # Linear layer
        out = self.lin(gru_out)

        # Add skip connection if it's enabled
        if self.use_skip:
            res_input = x.permute(0, 2, 1)
            res = self.res(res_input)
            out = out + res

        out = out.permute(0, 2, 1)  # output shape for loss: [batch, channel, seq]

        return out
