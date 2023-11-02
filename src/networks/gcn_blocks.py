import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Optional, Tuple, Union
from torch import Tensor


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, num_features: int) -> None:
        super().__init__()
        self.num_features = num_features
        self.adaptor = torch.nn.Linear(cond_dim, num_features * 2)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tensor:
        c = self.adaptor(c)
        g, b = torch.chunk(c, 2, dim=-1)
        g = g.permute(0, 2, 1)
        b = b.permute(0, 2, 1)
        x = (x * g) + b
        return x


class Conv1dCausal(nn.Module):
    """Causal convolution (padding applied to only left side)"""

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
            kernel_size,
            stride,
            padding=0,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, (self.padding, 0))  # standard zero padding
        x = self.conv(x)
        return x


class GatedConv1d(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, dilation: int, kernel_size: int, cond_dim: int
    ) -> None:
        super(GatedConv1d, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dilation = dilation
        self.kernel_size = kernel_size

        self.conv = Conv1dCausal(
            in_channels=in_ch,
            out_channels=out_ch * 2,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
        )

        self.film = FiLM(cond_dim=cond_dim, num_features=out_ch)

        self.mix = torch.nn.Conv1d(
            in_channels=out_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[Tensor, Tensor]:
        residual = x

        y = self.conv(x)

        # Gated activation function
        z = torch.tanh(y[:, : self.out_ch, :]) * torch.sigmoid(y[:, self.out_ch :, :])

        # Zero pad the left side, so that z is the same length as x
        zeros_tensor = torch.zeros(
            residual.size(0), self.out_ch, residual.size(2) - z.size(2), device=z.device
        )
        z = torch.cat((zeros_tensor, z), dim=2)

        z = self.film(z, c)

        x = self.mix(z)
        x = x + residual

        return x, z


class GCNBlock(nn.Module):
    def __init__(
        self, in_ch, out_ch, n_layers, kernel_size, dilation_growth, cond_dim
    ) -> None:
        super(GCNBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth
        self.cond_dim = cond_dim

        dilations = [dilation_growth**l for l in range(n_layers)]

        self.layers = nn.ModuleList()

        for d in dilations:
            self.layers.append(
                GatedConv1d(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    dilation=d,
                    kernel_size=kernel_size,
                    cond_dim=cond_dim,
                )
            )

            in_ch = out_ch

    def forward(self, x: Tensor, c: Tensor) -> Tuple[Tensor, Tensor]:
        z = torch.empty(
            [x.shape[0], self.n_layers * self.out_ch, x.shape[2]], device=x.device
        )

        for n, layer in enumerate(self.layers):
            x, z_n = layer(x, c)
            z[:, n * self.out_ch : (n + 1) * self.out_ch, :] = z_n

        return x, z


class GCN(nn.Module):
    def __init__(
        self,
        n_blocks: int = 1,
        n_layers: int = 10,
        n_channels: int = 8,
        kernel_size: int = 3,
        dilation_growth: int = 2,
        cond_dim: int = 3,
    ) -> Tensor:
        super(GCN, self).__init__()
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth
        self.cond_dim = cond_dim

        self.blocks = torch.nn.ModuleList()
        for b in range(n_blocks):
            self.blocks.append(
                GCNBlock(
                    in_ch=1 if b == 0 else n_channels,
                    out_ch=n_channels,
                    n_layers=n_layers,
                    kernel_size=kernel_size,
                    dilation_growth=dilation_growth,
                    cond_dim=cond_dim,
                )
            )

        # output mixing layer
        self.blocks.append(
            torch.nn.Conv1d(
                in_channels=n_channels * n_layers * n_blocks,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tensor:
        # x.shape = [batch, channels, length]

        z = torch.empty(
            [x.shape[0], self.blocks[-1].in_channels, x.shape[2]], device=x.device
        )

        for n, b in enumerate(self.blocks[:-1]):
            block = b
            x, z_n = block(x, c)
            z[
                :,
                n
                * self.n_channels
                * self.n_layers : (n + 1)
                * self.n_channels
                * self.n_layers,
                :,
            ] = z_n

        x = self.blocks[-1](z)

        return x

    def compute_receptive_field(self):
        RF = 1
        for _ in range(self.num_blocks):
            for l in range(self.num_layers):
                dilation = self.dilation_depth**l
                RF += (self.kernel_size - 1) * dilation
        return RF
