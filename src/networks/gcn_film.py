import torch
import torch.nn as nn

class FiLM(torch.nn.Module):
    def __init__(
        self,
        cond_dim,  # dim of conditioning input
        num_features,  # dim of the conv channel
        batch_norm=True,
    ):
        super().__init__()
        self.num_features = num_features
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = torch.nn.BatchNorm1d(num_features, affine=False)
        self.adaptor = torch.nn.Linear(cond_dim, num_features * 2)

    def forward(self, x, c):
        c = self.adaptor(c)
        g, b = torch.chunk(c, 2, dim=-1)
        g = g.permute(0, 2, 1)
        b = b.permute(0, 2, 1)
        
        if self.batch_norm:
            x = self.bn(x)      # apply BatchNorm without affine
        x = (x * g) + b     # then apply conditional affine

        return x
    

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
        
        # Ensure the zero tensor is on the same device as z
        zeros_tensor = torch.zeros(residual.size(0), self.out_channels, residual.size(2) - z.size(2), device=z.device)
        
        z = torch.cat((zeros_tensor, z), dim=2)
        return self.mix(z) + residual, z


class GCN_FiLM(torch.nn.Module):
    def __init__(self, num_blocks=1, num_layers=10, num_channels=8, kernel_size=3, dilation_depth=2, cond_dim=0):
        super(GCN_FiLM, self).__init__()
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.dilation_depth = dilation_depth
        self.cond_dim = cond_dim

        # Create the entire dilation sequence here
        dilations = [dilation_depth ** l for l in range(num_layers)] * num_blocks
        in_channels_sequence = [1] + [num_channels] * (len(dilations) - 1)
        
        # Initialize FiLM layer
        self.film_layer = FiLM(cond_dim=cond_dim, num_features=num_channels)
        
        # Gated Convolutional Layers
        self.layers = nn.ModuleList()
        for in_ch, d in zip(in_channels_sequence, dilations):
            self.layers.append(GatedConv1d(in_ch, num_channels, d, kernel_size))
            if cond_dim > 0:
                self.layers.append(self.film_layer)
                
        # output mixing layer
        self.mix = nn.Conv1d(in_channels=num_channels * num_layers * num_blocks, out_channels=1, kernel_size=1)

    def forward(self, x, c=None):
        z_storage = torch.empty([x.shape[0], self.mix.in_channels, x.shape[2]]).to(x.device)
        film_index = 0
        z_index = 0

        for n, layer in enumerate(self.layers):
            if isinstance(layer, GatedConv1d):
                x, zn = layer(x)
                z_storage[:, z_index*self.num_channels:(z_index+1)*self.num_channels, :] = zn
                z_index += 1
            elif isinstance(layer, FiLM):
                x = layer(x, c)
                film_index += 1

            # z_storage[:, n*self.num_channels:(n+1)*self.num_channels, :] = zn
        
        return self.mix(z_storage)
    
    def compute_receptive_field(self):
        RF = 1
        for _ in range(self.num_blocks):
            for l in range(self.num_layers):
                dilation = self.dilation_depth ** l
                RF += (self.kernel_size - 1) * dilation
        return RF
