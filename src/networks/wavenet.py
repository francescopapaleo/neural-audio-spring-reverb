import torch
from torch import nn

class CausalConv1d(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result

class WaveNetFF(nn.Module):
    """Feed-forward WaveNEt: a causal feedforward model, as such each output sample predicted by the model, depends only on the N previous input samples. 
    The receptive field depends on the number of convolutional layers, and the lengths of the filters in the layers. 
    with residual connections"""

    def __init__(self, num_channels=8, dilation_depth=4, num_layers=10, kernel_size=9):
        super(WaveNetFF, self).__init__()
        dilations = [2 ** d for d in range(dilation_depth)] * num_layers

        self.layers = nn.ModuleList([CausalConv1d(num_channels if i > 0 else 1, num_channels, kernel_size, dilation=d) for i, d in enumerate(dilations)])
        self.prelus = nn.ModuleList([nn.PReLU() for _ in dilations])

        self.linear_mix = nn.Conv1d(in_channels=num_channels *len(self.layers), out_channels=1, kernel_size=1)

    def forward(self, x):
        out_list = [x]

        for layer, prelu in zip(self.layers, self.prelus):
            out = layer(out_list[-1])  # Use the last output as the input for the next layer
            out = prelu(out)
            out_list.append(out)

        # Discard the original x from the list
        out_list = out_list[1:]
        
        # Concatenate outputs for the linear mixer
        out_combined = torch.cat(out_list, dim=1)

        # Use the 1x1 convolution as linear mixer
        out_mixed = self.linear_mix(out_combined)

        return out_mixed
    
    def compute_receptive_field(self):
        # Initialize receptive field with the kernel size of the first layer
        total_rf = self.layers[0].kernel_size[0]
        
        for layer in self.layers[1:]:
            dilation = layer.dilation[0]
            kernel_size = layer.kernel_size[0]
            
            # Update the receptive field for the current layer
            layer_rf = kernel_size + (kernel_size - 1) * (dilation - 1)
            
            # Update the total receptive field
            total_rf += layer_rf - 1

        return total_rf

