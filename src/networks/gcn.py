import torch
import torch.nn as nn

class GatedConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dilation, kernel_size):
        super(GatedConv1d, self).__init__()
        self.out_channels = out_channels

        self.conv = torch.nn.Conv1d(in_channels=in_channels,
                                    out_channels=out_channels * 2,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=0,
                                    dilation=dilation)
        self.mix = torch.nn.Conv1d(in_channels=out_channels,
                                   out_channels=out_channels,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)

    def forward(self, x):
        residual = x
        y = self.conv(x)
        z = torch.tanh(y[:, :self.out_channels, :]) * torch.sigmoid(y[:, self.out_channels:, :])
        z = torch.cat((torch.zeros(residual.size(0), self.out_channels, residual.size(2) - z.size(2)), z), dim=2)
        return self.mix(z) + residual, z


class GCN(torch.nn.Module):
    def __init__(self, num_blocks=2, num_layer=9, num_channels=8, kernel_size=3, dilation_depth=2):
        super(GCN, self).__init__()
        self.num_layer = num_layer
        self.num_channels = num_channels

        # Create the entire dilation sequence here
        dilations = [dilation_depth ** l for l in range(num_layer)] * num_blocks
        in_channels_sequence = [1] + [num_channels] * (len(dilations) - 1)
        self.layers = nn.ModuleList([GatedConv1d(in_ch, num_channels, d, kernel_size) for in_ch, d in zip(in_channels_sequence, dilations)])

        # output mixing layer
        self.mix = nn.Conv1d(in_channels=num_channels * num_layer * num_blocks, out_channels=1, kernel_size=1)

    def forward(self, x):
        # x = x.permute(1, 2, 0)
        z_storage = torch.empty([x.shape[0], self.mix.in_channels, x.shape[2]])
        for n, layer in enumerate(self.layers):
            x, zn = layer(x)
            z_storage[:, n*self.num_channels:(n+1)*self.num_channels, :] = zn

        # return self.mix(z_storage).permute(2, 0, 1)
        return self.mix(z_storage)
    
    def compute_receptive_field(kernel_size, dilation_depth, num_layer, num_blocks):
        RF = 1
        for _ in range(num_blocks):
            for l in range(num_layer):
                dilation = dilation_depth ** l
                RF += (kernel_size - 1) * dilation
        return RF
        