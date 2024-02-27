import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class FiLM(nn.Module):
    """Feature-wise Linear Modulation (FiLM) layer
    with batch normalization (BN) and affine transformation.

    Parameters:
        cond_dim (int): Dimension of the conditioning input.
        num_features (int): Number of feature maps in the input on which FiLM will be applied.

    Returns:
        Tensor: The output of the FiLM layer.
    """

    def __init__(
        self,
        cond_dim: int,  # dim of conditioning input
        n_features: int,  # dim of the conv channel
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = n_features
        self.adaptor = nn.Linear(cond_dim, n_features * 2)
        if batch_norm is True:
            self.bn = nn.BatchNorm1d(n_features)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        cond = self.adaptor(cond)
        g, b = torch.chunk(cond, 2, dim=-1)
        g = g.unsqueeze(-1)
        b = b.unsqueeze(-1)

        if hasattr(self, "bn"):
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

    Returns:
        Tensor: The output of the causal 1D convolutional layer.
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


class GatedAF(nn.Module):
    """Gated activation function
    applies a tanh activation to one half of the input
    and a sigmoid activation to the other half, and then multiplies them element-wise.

    Returns:
        Tensor: The output of the gated activation function.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        x_tanh, x_sigmoid = x.chunk(2, dim=1)  # Split the output into two halves

        x_tanh = torch.tanh(x_tanh)  # Apply tanh activation
        x_sigmoid = torch.sigmoid(x_sigmoid)  # Apply sigmoid activation

        # Element-wise multiplication of tanh and sigmoid activations
        x = x_tanh * x_sigmoid
        return x


class TanhAF(nn.Module):
    """Tanh activation function

    Returns:
        Tensor: The output of the tanh activation function.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        x = torch.tanh(x)
        return x
