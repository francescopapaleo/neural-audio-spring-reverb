# Code from tcn_bare and steerable_nafx with modifications

import torch
from utils.utils import causal_crop

def center_crop(x: torch.Tensor, shape: int) -> torch.Tensor:
    start = (x.size(-1)-shape)//2
    stop  = start + shape
    return x[...,start:stop]

class FiLM(torch.nn.Module):
    def __init__(self, 
                 num_features, 
                 cond_dim):
        super(FiLM, self).__init__()
        self.num_features = num_features
        self.bn = torch.nn.BatchNorm1d(num_features, affine=False)
        self.adaptor = torch.nn.Linear(cond_dim, num_features * 2)

    def forward(self, x, cond):

        cond = self.adaptor(cond)
        g, b = torch.chunk(cond, 2, dim=-1)
        
        g = g.permute(0,2,1)
        b = b.permute(0,2,1)
        
        x = self.bn(x)      # apply BatchNorm without affine
        x = (x * g) + b     # then apply conditional affine

        return x

class TCNBlock(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 dilation, 
                 cond_dim=0, 
                 activation=True):
        super().__init__()
        self.conv = torch.nn.Conv1d(
        in_channels, 
        out_channels, 
        kernel_size, 
        dilation=dilation, 
        padding=0, #((kernel_size-1)//2)*dilation,
        bias=True)
        if cond_dim > 0:
            self.film = FiLM(cond_dim, out_channels, batch_norm=False)
        if activation:
            #self.act = torch.nn.Tanh()
            self.act = torch.nn.PReLU()
        self.res = torch.nn.Conv1d(in_channels, out_channels, 1, bias=False)

    def forward(self, x, c=None):
        x_in = x
        x = self.conv(x)
        if hasattr(self, "film"):
            x = self.film(x, c)
        if hasattr(self, "act"):
            x = self.act(x)
        x_res = causal_crop(self.res(x_in), x.shape[-1])
        x = x + x_res

        return x


class TCN(torch.nn.Module):
    """ Temporal convolutional network with conditioning module.

        Args:
            nparams (int): Number of conditioning parameters.
            ninputs (int): Number of input channels (mono = 1, stereo 2). Default: 1
            noutputs (int): Number of output channels (mono = 1, stereo 2). Default: 1
            nblocks (int): Number of total TCN blocks. Default: 10
            kernel_size (int): Width of the convolutional kernels. Default: 3
            dialation_growth (int): Compute the dilation factor at each block as dilation_growth ** (n % stack_size). Default: 1
            channel_growth (int): Compute the output channels at each black as in_ch * channel_growth. Default: 2
            channel_width (int): When channel_growth = 1 all blocks use convolutions with this many channels. Default: 64
            stack_size (int): Number of blocks that constitute a single stack of blocks. Default: 10
            num_examples (int): Number of evaluation audio examples to log after each epochs. Default: 4
        """
    def __init__(self, 
                 nparams,
                 ninputs=1,
                 noutputs=1,
                 nblocks=10, 
                 kernel_size=13,
                 n_channels=64, 
                 dilation_growth=8, 
                 channel_growth=1, 
                 channel_width=32, 
                 stack_size=10,
                 num_examples=4,
                 save_dir=None,
                 **kwargs):
        super(TCN, self).__init__()

        self.nparams=nparams
        self.ninputs=ninputs
        self.noutputs=noutputs
        self.nblocks=nblocks
        self.kernel_size=kernel_size
        self.dilation_growth=dilation_growth
        self.channel_growth=channel_growth
        self.channel_width=channel_width
        self.stack_size=stack_size
        self.num_examples=num_examples
        self.save_dir=save_dir

        # setup loss functions
        self.l1      = torch.nn.L1Loss()

        if self.nparams > 0:
            self.gen = torch.nn.Sequential(
                torch.nn.Linear(nparams, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 32),
                torch.nn.ReLU()
            )

        self.blocks = torch.nn.ModuleList()
        for n in range(nblocks):
            in_ch = out_ch if n > 0 else ninputs
            
            if self.channel_growth > 1:
                out_ch = in_ch * self.channel_growth 
            else:
                out_ch = self.channel_width

            dilation = self.dilation_growth ** n
            self.blocks.append(TCNBlock(in_ch, 
                                        out_ch, 
                                        kernel_size=self.kernel_size, 
                                        dilation=dilation))

        self.output = torch.nn.Conv1d(out_ch, noutputs, kernel_size=1)

    # iterate over blocks passing conditioning
    def forward(self, x, c=None):
        for block in self.blocks:
            x = block(x, c)
        return x

    def compute_receptive_field(self):
        """ Compute the receptive field in samples."""
        rf = self.kernel_size
        for n in range(1,self.nblocks):
            dilation = self.dilation_growth ** (n % self.stack_size)
            rf = rf + ((self.kernel_size-1) * dilation)
        return rf

