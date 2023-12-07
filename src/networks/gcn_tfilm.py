import torch
import torch.nn as nn

from torch import Tensor, Optional
from src.networks.custom_layers import Conv1dCausal, GatedAF, TanhAF


class TFiLM(torch.nn.Module):
    def __init__(self, n_channels, n_params, tfilm_block_size, rnn_type="lstm"):
        super().__init__()
        self.nchannels = n_channels
        self.nparams = n_params
        self.tfilm_block_size = tfilm_block_size
        self.num_layers = 1
        self.first_run = True
        self.hidden_state = (
            torch.Tensor(0),
            torch.Tensor(0),
        )  # (hidden_state, cell_state)

        # to downsample input
        self.maxpool = torch.nn.MaxPool1d(
            kernel_size=tfilm_block_size,
            stride=None,
            padding=0,
            dilation=1,
            return_indices=False,
            ceil_mode=False,
        )

        if rnn_type.lower() == "lstm":
            self.rnn = torch.nn.LSTM(
                input_size=n_channels + n_params,
                hidden_size=n_channels,
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=False,
            )
        elif rnn_type.lower() == "gru":
            self.rnn = torch.nn.GRU(
                input_size=n_channels + n_params,
                hidden_size=n_channels,
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=False,
            )
        else:
            raise ValueError("Invalid rnn_type. Use 'lstm' or 'gru'.")

    def forward(self, x, p: Optional[Tensor] = None):
        # x = [batch_size, n_channels, length]
        # p = [batch_size, n_params]
        x_in_shape = x.shape

        # pad input if it's not multiple of tfilm block size
        if (x_in_shape[2] % self.tfilm_block_size) != 0:
            padding = torch.zeros(
                x_in_shape[0],
                x_in_shape[1],
                self.tfilm_block_size - (x_in_shape[2] % self.tfilm_block_size),
            )
            x = torch.cat((x, padding), dim=-1)

        x_shape = x.shape
        nsteps = int(x_shape[-1] / self.tfilm_block_size)

        # downsample signal [batch, nchannels, nsteps]
        x_down = self.maxpool(x)

        if self.nparams > 0 and p is not None:
            p_up = p.unsqueeze(-1)
            p_up = p_up.repeat(1, 1, nsteps)  # upsample params [batch, nparams, nsteps]
            x_down = torch.cat(
                (x_down, p_up), dim=1
            )  # concat along channel dim [batch, nchannels+nparams, nsteps]

        # shape for LSTM (length, batch, channels)
        x_down = x_down.permute(2, 0, 1)
        print(x_down.shape)

        # modulation sequence
        if self.first_run:  # state was reset
            x_norm, self.hidden_state = self.rnn(x_down, None)
            self.first_run = False
        else:
            x_norm, self.hidden_state = self.rnn(x_down, self.hidden_state)

        # put shape back (batch, channels, length)
        x_norm = x_norm.permute(1, 2, 0)

        # reshape input and modulation sequence into blocks
        x_in = torch.reshape(
            x, shape=(-1, self.nchannels, nsteps, self.tfilm_block_size)
        )
        x_norm = torch.reshape(x_norm, shape=(-1, self.nchannels, nsteps, 1))

        # multiply
        x_out = x_norm * x_in

        # return to original (padded) shape
        x_out = torch.reshape(x_out, shape=(x_shape))

        # crop to original (input) shape
        x_out = x_out[..., : x_in_shape[2]]

        return x_out

    def reset_state(self):
        self.first_run = True


class GCNBlock(nn.Module):
    """Single block of a Gated Convolutional Network (GCN) with conditional modulation.

    Parameters:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        kernel_size (int, optional): Size of the convolution kernel.
        dilation (int, optional): Dilation rate for dilated convolutions.
        stride (int, optional): Stride for the convolution.
        cond_dim (int, optional): Dimensionality of the conditional input for FiLM.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
        cond_dim: int = 0,
        rnn_type: str = "lstm",
        tfilm_block_size: int = 128,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.cond_dim = cond_dim
        self.rnn_type = rnn_type
        self.rnn_block_size = tfilm_block_size

        self.conv = Conv1dCausal(
            in_channels=in_ch,
            out_channels=out_ch * 2,  # adapt for the Gated Activation Function
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

        if cond_dim > 0:
            self.film = TFiLM(
                n_channels=out_ch * 2,
                n_params=cond_dim,
                tfilm_block_size=tfilm_block_size,
                rnn_type=self.rnn_type,
            )
        else:
            self.film = None

        self.gated_activation = GatedAF()

        self.res = nn.Conv1d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=(1,), bias=False
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        x_in = x
        x = self.conv(x)  # Apply causal convolution
        if self.film is not None:  # Apply FiLM if conditional input is given
            x = self.film(x, cond)
        x = self.gated_activation(x)  # Apply gated activation function
        x_res = self.res(x_in)  # Apply residual convolution
        x = x + x_res  # Apply residual connection
        return x


class GCNTFiLM(nn.Module):
    """Gated Convolutional Network (GCN) model, often used in sequence modeling tasks.

    Parameters:
        in_ch (int, optional): Number of input channels.
        out_ch (int, optional): Number of output channels.
        n_blocks (int, optional): Number of GCN blocks.
        n_channels (int, optional): Number of channels in the GCN blocks.
        dilation_growth (int, optional): Growth rate for dilation in the GCN blocks.
        kernel_size (int, optional): Size of the convolution kernel.
        cond_dim (int, optional): Dimensionality of the conditional input for FiLM.

    Returns:
        Tensor: The output of the GCN model.
    """

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        n_blocks: int = 2,
        n_channels: int = 32,
        dilation_growth: int = 8,
        kernel_size: int = 3,
        cond_dim: int = 3,
        tfilm_block_size: int = 128,
        rnn_type: str = "lstm",
    ) -> None:
        super().__init__()
        self.in_ch = in_ch  # input channels
        self.out_ch = out_ch  # output channels
        self.kernel_size = kernel_size
        self.cond_dim = cond_dim

        # Compute convolution channels and dilations
        self.channels = [n_channels] * n_blocks
        self.dilations = [dilation_growth**idx for idx in range(n_blocks)]
        print(f"Dilations: {self.dilations}")

        # Blocks number is given by the number of elements in the channels list
        self.n_blocks = len(self.channels)
        assert len(self.dilations) == self.n_blocks

        # Create a list of strides
        self.strides = [1] * self.n_blocks

        # Create a list of GCN blocks
        self.blocks = nn.ModuleList()
        block_out_ch = None

        for idx, (curr_out_ch, dil, stride) in enumerate(
            zip(self.channels, self.dilations, self.strides)
        ):
            if idx == 0:
                block_in_ch = in_ch
            else:
                block_in_ch = block_out_ch
            block_out_ch = curr_out_ch

            self.blocks.append(
                GCNBlock(
                    block_in_ch,
                    block_out_ch,
                    kernel_size,
                    dil,
                    stride,
                    cond_dim,
                    tfilm_block_size=tfilm_block_size,
                    rnn_type=rnn_type,
                )
            )

        # Output layer
        self.out_net = nn.Conv1d(
            self.channels[-1], out_ch, kernel_size=(1,), stride=(1,), bias=False
        )

        # Activation function
        self.af = TanhAF()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        # x.shape = (batch_size, in_ch, samples)
        # cond.shape = (batch_size, cond_dim)
        for block in self.blocks:  # Apply GCN blocks
            x = block(x, cond)
        x = self.out_net(x)  # Apply output layer
        x = self.af(x)  # Apply tanh activation function
        return x

    def calc_receptive_field(self) -> int:
        """Compute the receptive field in samples.

        Returns:
            int: The receptive field of the model.
        """
        assert all(_ == 1 for _ in self.strides)  # TODO(cm): add support for dsTCN
        assert self.dilations[0] == 1  # TODO(cm): add support for >1 starting dilation
        rf = self.kernel_size
        for dil in self.dilations[1:]:
            rf = rf + ((self.kernel_size - 1) * dil)
        return rf


if __name__ == "__main__":
    from torchinfo import summary

    model = GCNTFiLM(
        in_ch=1,
        out_ch=1,
        n_blocks=2,
        n_channels=32,
        dilation_growth=256,
        kernel_size=99,
        cond_dim=3,
        tfilm_block_size=128,
        rnn_type="lstm",
    )

    sample_rate = 48000

    model.eval()
    x = torch.randn(1, 1, 48000)
    cond = torch.randn(1, 3)

    summary(model, input_data=(x, cond), depth=4, verbose=1)
    rf = model.calc_receptive_field()
    print(f"Receptive field: {rf} samples or {(rf / sample_rate)*1e3:0.1f} ms")

    scripted_model = torch.jit.script(model)
    scripted_model.save("scripted_model.pt")
