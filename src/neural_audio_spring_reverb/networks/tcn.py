""" 
Temporal Convolutional Network (TCN) with FiLM conditioning module.

https://github.com/csteinmetz1/steerable-nafx/blob/main/steerable-nafx.ipynb

@inproceedings{steinmetz2021steerable,
    title={Steerable discovery of neural audio effects},
    author={Steinmetz, Christian J. and Reiss, Joshua D.},
    booktitle={5th Workshop on Creativity and Design at NeurIPS},
    year={2021}}
    """

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Optional, Tuple, Union
from torch import Tensor
from neural_audio_spring_reverb.networks.custom_layers import Conv1dCausal, FiLM


def center_crop(x, length: int):
    start = (x.shape[-1] - length) // 2
    stop = start + length
    return x[..., start:stop]


def causal_crop(x, length: int):
    if x.shape[-1] != length:
        stop = x.shape[-1] - 1
        start = stop - length
        x = x[..., start:stop]
    return x


class TCNBlock(torch.nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
        cond_dim: int = 0,
        activation=True,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.cond_dim = cond_dim

        self.conv = Conv1dCausal(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

        if cond_dim > 0:
            self.film = FiLM(cond_dim=cond_dim, n_features=out_ch, batch_norm=True)

        if activation:
            self.act = torch.nn.PReLU()

        self.res = nn.Conv1d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=(1,), bias=False
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        x_in = x
        x = self.conv(x)

        if hasattr(self, "film"):
            x = self.film(x, cond)

        if hasattr(self, "act"):
            x = self.act(x)

        x_res = causal_crop(self.res(x_in), x.shape[-1])
        x = x + x_res

        return x


class TCN(torch.nn.Module):
    """
    Temporal convolutional network with conditioning module.
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
    ):
        super().__init__()
        self.in_ch = in_ch  # input channels
        self.out_ch = out_ch  # output channels
        self.kernel_size = kernel_size
        self.cond_dim = cond_dim

        # Compute convolution channels and dilations
        self.channels = [n_channels] * n_layers
        self.dilations = [dilation_growth**idx for idx in range(n_layers)]

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
                TCNBlock(
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
        for block in self.blocks:
            x = block(x, cond)
        x = self.out_net(x)
        return x

    def calc_receptive_field(self):
        """Compute the receptive field in samples."""
        assert all(_ == 1 for _ in self.strides)  # TODO(cm): add support for dsTCN
        assert self.dilations[0] == 1  # TODO(cm): add support for >1 starting dilation
        rf = self.kernel_size
        for dil in self.dilations[1:]:
            rf = rf + ((self.kernel_size - 1) * dil)
        return rf
