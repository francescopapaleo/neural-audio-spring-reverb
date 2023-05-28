import torch
from utils.utils import center_crop, causal_crop
from argparse import ArgumentParser
'''
Code from : https://github.com/csteinmetz1/steerable-nafx (with some light modifications)
'''

class FiLM(torch.nn.Module):
    def __init__(
        self,
        cond_dim,       # dim of conditioning input
        num_features,   # dim of the conv channel
        batch_norm=False,
    ):
        super(FiLM).__init__()
        self.num_features = num_features
        self.bn = torch.nn.BatchNorm1d(num_features, affine=False)
        self.adaptor = torch.nn.Linear(cond_dim, num_features * 2)

    def forward(self, x, cond):

        cond = self.adaptor(cond)
        g, b = torch.chunk(cond, 2, dim=-1)
        g = g.permute(0,2,1)
        b = b.permute(0,2,1)

        if self.batch_norm:
            x = self.bn(x)  # apply BatchNorm without affine
        x = (x * g) + b     # then apply conditional affine

        return x

class TCNBlock(torch.nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, dilation, cond_dim=0, activation=True):
    super().__init__()
    self.conv = torch.nn.Conv1d(
        in_channels, 
        out_channels, 
        kernel_size, 
        dilation=dilation, 
        padding= ((kernel_size-1)//2)*dilation,     # uncommented this line
        bias=True)
    if cond_dim > 0:
      self.film = FiLM(cond_dim, out_channels, batch_norm=False)
    if activation:
      #self.act = torch.nn.Tanh()
      self.act = torch.nn.PReLU()
    self.res = torch.nn.Conv1d(in_channels, out_channels, 1, bias=False)

    def forward(self, x, p=None):
        x_in = x

        x = self.conv1(x)
        if self.grouped: # apply pointwise conv
           x = self.conv1b(x)
        if p is not None:   
          x = self.film(x, p) # apply FiLM conditioning
        else:
           x = self.bn(x)
        x = self.relu(x)

        x_res = self.res(x_in)
        if self.causal:
            x = x + causal_crop(x_res, x.shape[-1])
        else:
            x = x + center_crop(x_res, x.shape[-1])

        return x

class TCN(torch.nn.Module):
  def __init__(self, 
               n_inputs, 
               n_outputs, 
               n_blocks, 
               kernel_size, 
               n_channels, 
               dilation_growth,
               hparams,
               **kwargs):
    super().__init__()
    self.kernel_size = kernel_size
    self.n_channels = n_channels
    self.dilation_growth = dilation_growth
    self.n_blocks = n_blocks
    self.stack_size = n_blocks
    self.hparams = hparams              # added this line
    self.blocks = torch.nn.ModuleList()
    for n in range(n_blocks):
        in_ch = out_ch if n > 0 else n_inputs
        
        if self.hparams.channel_growth > 1:
            out_ch = in_ch * self.hparams.channel_growth 
        else:
            out_ch = self.hparams.channel_width

        dilation = self.hparams.dilation_growth ** (n % self.hparams.stack_size)
        self.blocks.append(TCNBlock(in_ch, 
                                    out_ch, 
                                    kernel_size=self.hparams.kernel_size, 
                                    dilation=dilation,
                                    padding="same" if self.hparams.causal else "valid",
                                    causal=self.hparams.causal,
                                    grouped=self.hparams.grouped,
                                    conditional=True if self.hparams.nparams > 0 else False))

    self.output = torch.nn.Conv1d(out_ch, n_outputs, kernel_size=1)

    def forward(self, x, p=None):
        # if parameters present, 
        # compute global conditioning
        if p is not None:
          cond = self.gen(p)
        else:
          cond = None

        # iterate over blocks passing conditioning
        for idx, block in enumerate(self.blocks):
            x = block(x, cond)
            if self.hparams.skip_connections:
               if idx == 0:
                   skips = x
               else:
                   skips = center_crop(skips, x[-1]) + x
            else:
              skips = 0

        out = torch.tanh(self.output(x + skips))

        return out

    def compute_receptive_field(self):
        """ Compute the receptive field in samples."""
        rf = self.hparams.kernel_size
        for n in range(1,self.hparams.nblocks):
            dilation = self.hparams.dilation_growth ** (n % self.hparams.stack_size)
            rf = rf + ((self.hparams.kernel_size-1) * dilation)
        return rf

    # add any model hyperparameters here
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # --- model related ---
        parser.add_argument('--ninputs', type=int, default=1)
        parser.add_argument('--noutputs', type=int, default=1)
        parser.add_argument('--nblocks', type=int, default=4)
        parser.add_argument('--kernel_size', type=int, default=5)
        parser.add_argument('--dilation_growth', type=int, default=10)
        parser.add_argument('--channel_growth', type=int, default=1)
        parser.add_argument('--channel_width', type=int, default=32)
        parser.add_argument('--stack_size', type=int, default=10)
        parser.add_argument('--grouped', default=False, action='store_true')
        parser.add_argument('--causal', default=False, action="store_true")
        parser.add_argument('--skip_connections', default=False, action="store_true")

        return parser