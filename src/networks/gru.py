import torch
import torch.nn as nn

from src.networks.custom_layers import Conv1dCausal, FiLM, TanhAF

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

        # assert cond_dim > 0, "cond_dim must be greater than 0 for FiLM layer."

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.output_size = output_size
        self.dropout_prob = dropout_prob
        self.use_skip = use_skip
        self.kernel_size = kernel_size
        self.cond_dim = cond_dim

        # Convolutional layer and batch normalization
        self.conv1d = Conv1dCausal(
            self.input_size,
            self.hidden_size,
            self.kernel_size,
            stride=1,
            dilation=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(self.hidden_size)
        self.prelu1 = nn.PReLU()

        # GRU layer
        self.gru = nn.GRU(
            self.hidden_size,
            self.hidden_size,
            num_layers=self.n_layers,
            batch_first=True,
            dropout=self.dropout_prob if self.n_layers > 1 else 0,
        )

        # FiLM layer
        self.film_layer = FiLM(cond_dim=self.cond_dim, n_features=self.hidden_size)

        # Optional skip connection
        if self.use_skip:
            self.res = nn.Conv1d(
                in_channels=input_size, out_channels=output_size, kernel_size=(1,), bias=False
            )

        # Output layer
        self.out_net = nn.Conv1d(
            self.hidden_size, output_size, kernel_size=(1,), stride=(1,), bias=False
        )

        # Activation function
        self.af = TanhAF()

    def forward(self, x, c):
        if c is None:
            raise ValueError("Conditional input 'c' is required for FiLM layer.")

        # Feature extraction
        x_features = self.conv1d(x)
        x_features = self.bn1(x_features)
        x_features = self.prelu1(x_features)

        # GRU layer
        x_permuted = x_features.permute(0, 2, 1)
        gru_out, _ = self.gru(x_permuted)
        out = gru_out.permute(0, 2, 1)

        # Apply FiLM modulation
        out = self.film_layer(out, c)
        
        # Add skip connection if it's enabled
        if self.use_skip:
            res = self.res(x)
            out = out + res

        # Linear layer
        out = self.out_net(out)

        # Apply activation function
        out = self.af(out)

        return out


if __name__ == "__main__":


    from torchinfo import summary

    # Test GRU
    model = GRU(
        input_size=1,
        hidden_size=64,
        n_layers=2,
        output_size=1,
        dropout_prob=0.1,
        use_skip=True,
        kernel_size=3,
        cond_dim=3,
    )

    sample_rate = 48000

    model.eval()
    x = torch.randn(1, 1, 48000)
    cond = torch.randn(1, 3)

    summary(model, input_data=(x, cond), depth=4, verbose=2)