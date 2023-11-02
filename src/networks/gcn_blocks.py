import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Optional, Tuple, Union
from torch import Tensor


class FiLM(nn.Module):
    """Feature-wise Linear Modulation (FiLM) layer 
    with batch normalization (BN) and affine transformation.

    Parameters:
        cond_dim (int): Dimension of the conditioning input.
        num_features (int): Number of feature maps in the input on which FiLM will be applied.
    """

    def __init__(
        self,
        cond_dim: int,  # dim of conditioning input
        num_features: int,  # dim of the conv channel
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.adaptor = nn.Linear(cond_dim, num_features * 2)
        self.bn = nn.BatchNorm1d(num_features)
        
    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        cond = self.adaptor(cond)
        g, b = torch.chunk(cond, 2, dim=-1)
        g = g.unsqueeze(-1)
        b = b.unsqueeze(-1)

        x = self.bn(x)

        x = (x * g) + b  # Then apply conditional affine
        return x


class Conv1dCausal(nn.Module):  # Conv1d with cache
    """Causal 1D convolutional layer
    ensures outputs depend only on current and past inputs.
    
    Parameters:
        in_channels (int): Number of channels in the input signal.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution.
        dilation (int, optional): Spacing between kernel elements.
        bias (bool, optional): If True, adds a learnable bias to the output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.padding = (
            kernel_size - 1
        ) * dilation  # input_len == output_len when stride=1
        self.in_channels = in_channels
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            (kernel_size,),
            (stride,),
            padding=0,
            dilation=(dilation,),
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, (self.padding, 0))  # standard zero padding
        x = self.conv(x)
        return x


class GatedActivation(nn.Module):
    """Gated Activation Function
    applies a tanh activation to one half of the input
    and a sigmoid activation to the other half, and then multiplies them element-wise.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        x_tanh, x_sigmoid = x.chunk(2, dim=1)  # Split the output into two halves

        x_tanh = torch.tanh(x_tanh)  # Apply tanh activation
        x_sigmoid = torch.sigmoid(x_sigmoid)  # Apply sigmoid activation

        # Element-wise multiplication of tanh and sigmoid activations
        gated_x = x_tanh * x_sigmoid
        return gated_x


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
            out_channels=out_ch * 2,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )
        assert cond_dim > 0
        self.film = FiLM(cond_dim=cond_dim, num_features=out_ch * 2)

        self.gated_activation = GatedActivation()

        self.res = nn.Conv1d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=(1,), bias=False
        )
        

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        x_in = x
        x = self.conv(x)  # Apply gated convolution
        if self.film is not None:
            x = self.film(
                x, cond
            )  # Apply FiLM layer if conditional dimension is provided
        x = self.gated_activation(x)
        x_res = self.res(x_in)
        x += x_res  # Apply residual connection
        return x


class GCN(nn.Module):
    """Gated Convolutional Network (GCN) model, often used in sequence modeling tasks.
    
    Parameters:
        n_channels (int): Number of channels for each hidden layer in the network.
        n_layers (int): Number of layers in the GCN.
        dilation_growth (int): Factor by which the dilation rate increases with each layer.
        in_ch (int, optional): Number of input channels.
        out_ch (int, optional): Number of output channels.
        kernel_size (int, optional): Size of the convolution kernel.
        cond_dim (int, optional): Dimensionality of the conditional input for FiLM.

    The network consists of a series of GCN blocks, where each block's dilation rate grows
    exponentially with the dilation growth factor, enabling the network to have a large receptive field.
    """
    def __init__(
        self,
        n_channels: int,
        n_layers: int,
        dilation_growth: int,
        in_ch: int = 1,
        out_ch: int = 1,
        kernel_size: int = 3,
        cond_dim: int = 0,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch  # input channels
        self.out_ch = out_ch    # output channels
        self.kernel_size = kernel_size
        self.cond_dim = cond_dim

        # Compute convolution channels and dilations
        self.channels = [n_channels] * n_layers
        self.dilations = [dilation_growth ** idx for idx in range(n_layers)]

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

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        assert x.ndim == 3  # (batch_size, in_ch, samples)
        assert cond.shape == (x.size(0), self.cond_dim)  # (batch_size, cond_dim)
        for block in self.blocks:
            x = block(x, cond)
        x = self.out_net(x)
        return x

    def compute_receptive_field(self):
        RF = 1
        for _ in range(self.num_blocks):
            for l in range(self.num_layers):
                dilation = self.dilation_depth**l
                RF += (self.kernel_size - 1) * dilation
        return RF
