import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Dict, List, Optional, Tuple, Union
from src.networks.custom_layers import Conv1dCausal, GatedAF, TanhAF, FiLM


class Conv1dStack(nn.Module):
    """
    Parameters:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        dilation (int): Spacing between kernel elements.
        cond_dim (int): Dimensionality of the conditional input for FiLM.

    Returns:
        Tensor: The output of the stack.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        cond_dim: int,
    ) -> None:
        super().__init__()

        # Causal convolutional layer
        self.conv = Conv1dCausal(
            in_channels=in_ch,
            out_channels=out_ch * 2,  # adapt for the GatedActivation
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
        )

        # FiLM layer
        self.film = FiLM(cond_dim=cond_dim, n_features=out_ch * 2)

        # Gated activation function
        self.gated_activation = GatedAF()

        self.res = nn.Conv1d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=(1,), bias=False
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        x_in = x
        x = self.conv(x)  # Apply causal convolution
        x = self.film(x, cond)  # Apply FiLM
        x = self.gated_activation(x)  # Apply gated activation function
        x_res = self.res(x_in)  # Apply residual convolution
        x = x + x_res  # Apply residual connection
        return x


class WaveNet1dBlock(nn.Module):
    """
    Parameters:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        n_stacks (int): Number of stacks in the block.
        kernel_size (int): Size of the convolution kernel.
        dilation_growth (int): Dilation growth rate for dilated convolutions.
        cond_dim (int): Dimensionality of the conditional input for FiLM.

    Returns:
        Tensor: The output of the block.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        n_stacks: int,
        kernel_size: int,
        dilation_growth: int,
        cond_dim: int,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.n_stacks = n_stacks
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth
        self.cond_dim = cond_dim

        dilations = [dilation_growth**s for s in range(n_stacks)]

        # Initialize the stacks of the block
        self.stacks = nn.ModuleList()

        for d in dilations:
            self.stacks.append(
                Conv1dStack(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    kernel_size=kernel_size,
                    dilation=d,
                    cond_dim=cond_dim,
                )
            )
            in_ch = out_ch  # Update the number of input channels

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        # x: (batch_size, in_ch, seq_len)
        for stack in self.stacks:
            x = stack(x, cond)

        return x


class WaveNet(nn.Module):
    """
    Parameters:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        n_blocks (int): Number of blocks in the model.
        n_stacks (int): Number of stacks in each block.
        n_channels (int): Number of channels in the hidden layers.
        kernel_size (int): Size of the convolution kernel.
        dilation_growth (int): Dilation growth rate for dilated convolutions.
        cond_dim (int): Dimensionality of the conditional input for FiLM.

    Returns:
        Tensor: The output of the model.
    """

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        n_blocks: int = 2,
        n_stacks: int = 2,
        n_channels: int = 32,
        kernel_size: int = 3,
        dilation_growth: int = 8,
        cond_dim: int = 3,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.n_blocks = n_blocks
        self.n_stacks = n_stacks
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth
        self.cond_dim = cond_dim

        # Standard list that holds the blocks of the model
        self.blocks = nn.ModuleList()

        for b in range(n_blocks):
            self.blocks.append(
                WaveNet1dBlock(
                    in_ch=in_ch if b == 0 else n_channels,
                    out_ch=n_channels,
                    n_stacks=n_stacks,
                    kernel_size=kernel_size,
                    dilation_growth=dilation_growth,
                    cond_dim=cond_dim,
                )
            )

        # Output linear mixing layer
        self.out_net = nn.Conv1d(
            in_channels=n_channels,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        # Activation function
        self.af = TanhAF()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        # x: (batch_size, in_ch, seq_len)
        # p: (batch_size, cond_dim)

        for block in self.blocks:
            x = block(x, cond)
        x = self.out_net(x)  # Apply output block
        x = self.af(x)  # Apply activation function
        return x

    def calc_receptive_field(self) -> int:
        """Calculate the receptive field of the model.
        The receptive field is the number of input samples that affect the output of a layer.

        The receptive field of a layer (except the first) is calculated as follows:
        RF_{\text{new}} = RF_{\text{prev}} + (kernel\_size - 1) \cdot dilation

        The receptive field of the model is the sum of the receptive fields of all layers:
        RF = 1 + \sum_{i=1}^{n}(kernel\_size_i - 1) \cdot dilation_i

        i is the layer index, n is the number of layers.

        Returns:
            int: The receptive field of the model.
        """
        receptive_field = 1
        for b in range(self.n_blocks):
            for s in range(self.n_stacks):
                dilation = self.dilation_growth**s
                receptive_field += (self.kernel_size - 1) * dilation
        return receptive_field


if __name__ == "__main__":
    from torchinfo import summary

    # Test Model
    print("\nTest Model")
    model = WaveNet()

    sample_rate = 48000

    model.eval()
    x = torch.randn(1, 1, 48000)
    cond = torch.randn(1, 3)

    # Print model summary
    summary(model, input_data=[x, cond], depth=6)
    rf = model.calc_receptive_field()
    print(f"Receptive field: {rf} samples or {(rf / sample_rate)*1e3:0.1f} ms")

    scripted_model = torch.jit.script(model)
    scripted_model.save("scripted_model.pt")
