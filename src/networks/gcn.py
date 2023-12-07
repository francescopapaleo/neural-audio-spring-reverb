import torch
import torch.nn as nn

from torch import Tensor
from src.networks.custom_layers import Conv1dCausal, FiLM, GatedAF, TanhAF


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
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.cond_dim = cond_dim

        self.conv = Conv1dCausal(
            in_channels=in_ch,
            out_channels=out_ch * 2,  # adapt for the Gated Activation Function
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

        self.film = FiLM(cond_dim=cond_dim, n_features=out_ch * 2)

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


class GCN(nn.Module):
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

    model = GCN(
        in_ch=1,
        out_ch=1,
        n_blocks=2,
        n_channels=32,
        dilation_growth=256,
        kernel_size=99,
        cond_dim=3,
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
