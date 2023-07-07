import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, : -self.__padding]
        return result


def _conv_stack(dilations, in_channels, out_channels, kernel_size, cond_dim=None):
    """
    Create stack of dilated convolutional layers, outlined in WaveNet paper:
    https://arxiv.org/pdf/1609.03499.pdf
    """
    if cond_dim is not None:  # For ConditionedCausalConv1d
        return nn.ModuleList(
            [
                ConditionedCausalConv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dilation=d,
                    kernel_size=kernel_size,
                    cond_dim=cond_dim,
                )
                for i, d in enumerate(dilations)
            ]
        )
    else:  # For regular CausalConv1d
        return nn.ModuleList(
            [
                CausalConv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dilation=d,
                    kernel_size=kernel_size,
                )
                for i, d in enumerate(dilations)
            ]
        )


class WaveNet(nn.Module):
    def __init__(self, 
                 n_channels, 
                 dilation, 
                 num_repeat, 
                 kernel_size):
        super(WaveNet, self).__init__()

        self.n_channels = n_channels
        self.dilation = dilation
        self.num_repeat = num_repeat
        self.kernel_size = kernel_size

        dilations = [2 ** d for d in range(dilation)] * num_repeat
        internal_channels = int(n_channels * 2)
        self.hidden = _conv_stack(dilations, n_channels, internal_channels, kernel_size)
        self.residuals = _conv_stack(dilations, n_channels, n_channels, 1)
        self.input_layer = CausalConv1d(
            in_channels=1,
            out_channels=n_channels,
            kernel_size=1,
        )

        self.linear_mix = nn.Conv1d(
            in_channels=n_channels * dilation * num_repeat,
            out_channels=1,
            kernel_size=1,
        )
        self.n_channels = n_channels

    def forward(self, x, c=None):
        out = x
        skips = []
        out = self.input_layer(out)

        for hidden, residual in zip(self.hidden, self.residuals):
            x = out
            out_hidden = hidden(x)

            # gated activation
            # split (32,16,3) into two (16,16,3) for tanh and sigm calculations
            out_hidden_split = torch.split(out_hidden, self.n_channels, dim=1)
            out = torch.tanh(out_hidden_split[0]) * torch.sigmoid(out_hidden_split[1])

            skips.append(out)

            out = residual(out)
            out = out + x[:, :, -out.size(2) :]

        # modified "postprocess" step:
        out = torch.cat([s[:, :, -out.size(2) :] for s in skips], dim=1)
        out = self.linear_mix(out)
        return out

    def compute_receptive_field(self):
        # Compute the receptive field for each layer
        layers_rf = [self.kernel_size * (2 ** d) for d in range(self.dilation)] * self.num_repeat
        # The total receptive field is the sum of the receptive field of all layers
        total_rf = sum(layers_rf)
        return total_rf

class ConditionedCausalConv1d(CausalConv1d):
    def __init__(self, in_channels, out_channels, kernel_size, cond_dim, stride=1, dilation=1, groups=1, bias=True):
        super(ConditionedCausalConv1d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.cond_dim = cond_dim
        self.conditioning_layer = torch.nn.Conv1d(in_channels=cond_dim, out_channels=out_channels, kernel_size=1)

    def forward(self, input, conditions):
        result = super(ConditionedCausalConv1d, self).forward(input)
        condition_effect = condition_effect.squeeze(-1)
        
        # Expand condition_effect to match the size of result
        condition_effect = condition_effect.expand_as(result)
        
        result += condition_effect

        if self.__padding != 0:
            return result[:, :, : -self.__padding]
        return result


class ConditionedWaveNet(WaveNet):
    def __init__(self, n_channels, dilation, num_repeat, kernel_size, cond_dim):
        super(ConditionedWaveNet, self).__init__(n_channels, dilation, num_repeat, kernel_size)
        dilations = [2 ** d for d in range(dilation)] * num_repeat
        internal_channels = int(n_channels * 2)
        self.hidden = _conv_stack(dilations, n_channels, internal_channels, kernel_size, cond_dim)
        self.residuals = _conv_stack(dilations, n_channels, n_channels, 1)  # The residuals don't need conditioning


    def forward(self, x, c):
        out = x
        skips = []
        out = self.input_layer(out)

        for hidden, residual in zip(self.hidden, self.residuals):
            x = out
            out_hidden = hidden(x, c)  # Pass the conditioning input as well

            out_hidden_split = torch.split(out_hidden, self.n_channels, dim=1)
            out = torch.tanh(out_hidden_split[0]) * torch.sigmoid(out_hidden_split[1])

            skips.append(out)

            out = residual(out)
            out = out + x[:, :, -out.size(2) :]

        out = torch.cat([s[:, :, -out.size(2) :] for s in skips], dim=1)
        out = self.linear_mix(out)
        return out
